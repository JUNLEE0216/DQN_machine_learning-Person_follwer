#!/usr/bin/env python3

import collections
import datetime
import json
import math
import os
import random
import sys
import time

import numpy
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty
import tensorflow
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from turtlebot3_msgs.srv import Dqn

# TensorFlow GPU 사용 비활성화 (CPU 전용 실행)
tensorflow.config.set_visible_devices([], 'GPU')

LOGGING = True
current_time = datetime.datetime.now().strftime('[%mm%dd-%H:%M]')

# DQN 학습 중 텐서보드에 기록할 커스텀 Metric 클래스
class DQNMetric(tensorflow.keras.metrics.Metric):
    # 누적 loss와 step 수 저장
    def __init__(self, name='dqn_metric'):
        super(DQNMetric, self).__init__(name=name)
        self.loss = self.add_weight(name='loss', initializer='zeros')
        self.episode_step = self.add_weight(name='step', initializer='zeros')
    # Metric 상태 업데이트 (에피소드별 보상(reward)을 누적해 업데이트)
    def update_state(self, y_true, y_pred=0, sample_weight=None):
        self.loss.assign_add(y_true)
        self.episode_step.assign_add(1)
    # reward 평균 반환
    def result(self):
        return self.loss / self.episode_step
    # Metric 상태 초기화
    def reset_states(self):
        self.loss.assign(0)
        self.episode_step.assign(0)

# DQN 에이전트 노드 클래스
class DQNAgent(Node):
    # DQNAgent 노드 초기화 및 하이퍼파라미터 설정
    def __init__(self, stage_num, max_training_episodes):
        super().__init__('dqn_agent')

        self.stage = int(stage_num)
        self.train_mode = True
        self.state_size = 5
        self.action_size = 5
        self.max_training_episodes = int(max_training_episodes)

        self.done = False
        self.succeed = False
        self.fail = False

        self.discount_factor = 0.99
        self.learning_rate = 0.0007
        self.epsilon = 1.0
        self.step_counter = 0
        self.epsilon_decay = 10000 * self.stage # 뒤 코드에 설명이 나오지만 앱실론 값들은 매우중요함!
        self.epsilon_min = 0.05
        self.batch_size = 128

        self.replay_memory = collections.deque(maxlen=500000)
        self.min_replay_memory_size = 5000
        
        # Q-network 및 Target Q-network 생성
        self.model = self.create_qnetwork()
        self.target_model = self.create_qnetwork()
        self.update_target_model()
        self.update_target_after = 5000
        self.target_update_after_counter = 0

        # 기존 학습 모델 로드 설정
        self.load_model = True
        self.load_episode = 1000
        self.model_dir_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'saved_model'
        )
        self.model_path = os.path.join(
            self.model_dir_path,
            'stage' + str(self.stage) + '_episode' + str(self.load_episode) + '.h5'
        )
        # 저장된 모델 및 파라미터 로드
        if self.load_model:
            self.model.set_weights(load_model(self.model_path).get_weights())
            with open(os.path.join(
                self.model_dir_path,
                'stage' + str(self.stage) + '_episode' + str(self.load_episode) + '.json'
            )) as outfile:
                param = json.load(outfile)
                self.epsilon = param.get('epsilon', 1.0)
                if 'step_counter' in param:
                    self.step_counter = param['step_counter']
                elif 'step' in param:
                    self.step_counter = param['step']
                else:
                    self.step_counter = 0
        # TensorBoard 로그 설정
        if LOGGING:
            tensorboard_file_name = current_time + ' dqn_stage' + str(self.stage) + '_reward'
            home_dir = os.path.expanduser('~')
            dqn_reward_log_dir = os.path.join(
                home_dir, 'turtlebot3_dqn_logs', 'gradient_tape', tensorboard_file_name
            )
            self.dqn_reward_writer = tensorflow.summary.create_file_writer(dqn_reward_log_dir)
            self.dqn_reward_metric = DQNMetric()

        # ROS2 서비스 및 퍼블리셔 설정
        self.rl_agent_interface_client = self.create_client(Dqn, 'rl_agent_interface')
        self.make_environment_client = self.create_client(Empty, 'make_environment')
        self.reset_environment_client = self.create_client(Dqn, 'reset_environment')

        self.action_pub = self.create_publisher(Float32MultiArray, '/get_action', 10)
        self.result_pub = self.create_publisher(Float32MultiArray, 'result', 10)
        
        # 장애물 회피 관련 변수
        self.avoid_mode = False
        self.avoid_counter = 0
        self.AVOID_DIST = 0.35
        self.AVOID_RELEASE_DIST = 0.55
        self.MAX_AVOID_STEP = 8  # 회피 최대 스텝 수 - 1 스텝당 약 0.2초

        # 메인 학습 루프 실행
        self.process()
    
    #전체 학습 프로세스를 관리하는 메인 루프
    def process(self):
        self.env_make()
        time.sleep(1.0)

        episode_num = self.load_episode

        for episode in range(self.load_episode + 1, self.max_training_episodes + 1):
            state = self.reset_environment()
            episode_num += 1
            local_step = 0
            score = 0
            sum_max_q = 0.0

            

            while True:
                local_step += 1

                q_values = self.model.predict(state)
                sum_max_q += float(numpy.max(q_values))

                action = int(self.get_action(state))
                next_state, reward, done = self.step(action)
                score += reward

                msg = Float32MultiArray()
                msg.data = [float(action), float(score), float(reward)]
                self.action_pub.publish(msg)

                if self.train_mode:
                    self.append_sample((state, action, reward, next_state, done))
                    self.train_model(done)

                state = next_state

                if done:
                    avg_max_q = sum_max_q / local_step if local_step > 0 else 0.0

                    msg = Float32MultiArray()
                    msg.data = [float(score), float(avg_max_q)]
                    self.result_pub.publish(msg)

                    if LOGGING:
                        self.dqn_reward_metric.update_state(score)
                        with self.dqn_reward_writer.as_default():
                            tensorflow.summary.scalar(
                                'dqn_reward', self.dqn_reward_metric.result(), step=episode_num
                            )
                        self.dqn_reward_metric.reset_states()

                    print(
                        f'\n'
                        f'==============================\n'
                        f'Episode        : {episode}\n'
                        f'Score          : {score:.3f}\n'
                        f'Memory Length  : {len(self.replay_memory)}\n'
                        f'Epsilon        : {self.epsilon:.6f}\n'
                        f'==============================\n'
                    )


                    param_keys = ['epsilon', 'step_counter']
                    param_values = [self.epsilon, self.step_counter]
                    param_dictionary = dict(zip(param_keys, param_values))
                    break

                time.sleep(0.01)
            
            # 주기적으로 모델 저장
            if self.train_mode:
                if episode % 50 == 0:  # 50개 에피소드마다 저장됨
                    self.model_path = os.path.join(
                        self.model_dir_path,
                        'stage' + str(self.stage) + '_episode' + str(episode) + '.h5')
                    self.model.save(self.model_path)
                    with open(
                        os.path.join(
                            self.model_dir_path,
                            'stage' + str(self.stage) + '_episode' + str(episode) + '.json'
                        ),
                        'w'
                    ) as outfile:
                        json.dump(param_dictionary, outfile)
    # 가제보 환경 생성 함수
    def env_make(self):
        while not self.make_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                'Environment make client failed to connect to the server, try again ...'
            )

        self.make_environment_client.call_async(Empty.Request())

    # 환경 리셋 및 초기 상태 반환
    def reset_environment(self):
        while not self.reset_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                'Reset environment client failed to connect to the server, try again ...'
            )

        future = self.reset_environment_client.call_async(Dqn.Request())

        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            state = future.result().state
            state = numpy.reshape(numpy.asarray(state), [1, self.state_size])
        else:
            self.get_logger().error(
                'Exception while calling service: {0}'.format(future.exception()))

        return state

    # ε-greedy(행동 선택을 위한 탐색 전략) + 장애물 회피 기반 액션 선택(에피소드 초반 학습의 효율을 높여주기 위해)
    def get_action(self, state):
        # state는 환경으로부터 받은 관측값인데 여기서 라이다 센서 값 추출
        # state = [[ s0, s1, s2, s3, s4 ]]
        #           |   |   |   |   |
        #           |   |   |   |   └─ state[0][4]
        #           |   |   |   └──── state[0][3]
        #           |   |   └──────── state[0][2]
        #           |   └──────────── state[0][1]
        #           └──────────────── state[0][0]
        # 각각 좌측,우측,정면의 라이다 거리 값
        left = state[0][2]   # 2번째 관측 요소
        center = state[0][3] # 3번째 관측 요소
        right = state[0][4] # 4번째 관측 요소

        # train_mode = True   학습 중 (탐색 + 회피 + ε-greedy)
        # train_mode = False  테스트/실행 모드 (탐색 없음)
        if self.train_mode:
            # 1. 장애물 회피 모드 (avoid_mode) 동작 중일 때
            if self.avoid_mode:
                self.avoid_counter += 1 # 회피 동작마다 카운트 증가
                
                # 이건 회피 모드 해제 조건인데
                # 정면 거리가 충분히 멀어질때
                # 회피 동작이 일정 스텝 이상 진행됐을때 
                if center > self.AVOID_RELEASE_DIST and self.avoid_counter >= self.MAX_AVOID_STEP:
                    self.avoid_mode = False
                    self.avoid_counter = 0
                else:
                    # 아직 회피 중이면 좌/우 중에 더 넓은 쪽으로 이동
                    if left < right:
                        return 2
                    else:
                        return 1
            
            # 2. 정면에 장애물이 가까이 있을 경우  회피 모드 진입
            if center < self.AVOID_DIST:
                self.avoid_mode = True
                self.avoid_counter = 0

                # 좌/우 중 더 넓은 쪽으로 회피 방향 결정
                if left < right:
                    return 2   
                else:
                    return 1  
                
            # 3. ε-greedy 탐색을 위한 epsilon 값 업데이트
            self.step_counter += 1
            self.epsilon = self.epsilon_min + (1.0 - self.epsilon_min) * math.exp(
                -1.0 * self.step_counter / self.epsilon_decay
            )
            # 중요!! 학습이 진행될수록 앱실론 값이 점점 줄어들어 
            # 랜덤 행동 확률이 낮아지고 학습 기반 행동 비율이 증가함
            # 그래서 학습 초반에 앱실론 값을 천천히 줄이는 것이 매우 중요!!
            # 앱실론 값의 조절이 학습 성능에 큰 영향을 미친다

            if random.random() < self.epsilon:
                return random.randint(0, self.action_size - 1)
            else:
                return numpy.argmax(self.model.predict(state))

        else:
            return numpy.argmax(self.model.predict(state))
    
    # 환경에서 액션 수행 및 다음 상태, 보상, 종료 여부를 받는 함수
    def step(self, action):
        req = Dqn.Request()
        req.action = action

        while not self.rl_agent_interface_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('rl_agent interface service not available, waiting again...')

        future = self.rl_agent_interface_client.call_async(req) 

        rclpy.spin_until_future_complete(self, future)


        if future.result() is not None:
            next_state = future.result().state
            next_state = numpy.reshape(numpy.asarray(next_state), [1, self.state_size])
            reward = future.result().reward
            done = future.result().done
        else:
            self.get_logger().error(
                'Exception while calling service: {0}'.format(future.exception()))

        return next_state, reward, done

    # keras sequential 을 사용해 q-network 모델 생성하는 함수
    def create_qnetwork(self):
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        model.add(Dense(512, activation='relu')) 
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=self.learning_rate))
        model.summary()

        return model
    # 타겟 모델 업데이트 함수
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        self.target_update_after_counter = 0
        print('*Target model updated*')

    # 리플레이에 샘플 추가 함수
    def append_sample(self, transition):
        self.replay_memory.append(transition)

    # 미니배치를 사용해 모델 학습하는 함수
    def train_model(self, terminal):
        if len(self.replay_memory) < self.min_replay_memory_size:
            return
        data_in_mini_batch = random.sample(self.replay_memory, self.batch_size)

        current_states = numpy.array([transition[0] for transition in data_in_mini_batch])
        current_states = current_states.squeeze()
        current_qvalues_list = self.model.predict(current_states)

        next_states = numpy.array([transition[3] for transition in data_in_mini_batch])
        next_states = next_states.squeeze()
        next_qvalues_list = self.target_model.predict(next_states)

        x_train = []
        y_train = []

        for index, (current_state, action, reward, _, done) in enumerate(data_in_mini_batch):
            current_q_values = current_qvalues_list[index]

            if not done:
                future_reward = numpy.max(next_qvalues_list[index])
                desired_q = reward + self.discount_factor * future_reward
            else:
                desired_q = reward

            current_q_values[action] = desired_q
            x_train.append(current_state)
            y_train.append(current_q_values)

        x_train = numpy.array(x_train)
        y_train = numpy.array(y_train)
        x_train = numpy.reshape(x_train, [len(data_in_mini_batch), self.state_size])
        y_train = numpy.reshape(y_train, [len(data_in_mini_batch), self.action_size])

        self.model.fit(
            tensorflow.convert_to_tensor(x_train, tensorflow.float32),
            tensorflow.convert_to_tensor(y_train, tensorflow.float32),
            batch_size=self.batch_size, verbose=0
        )
        self.target_update_after_counter += 1

        if self.target_update_after_counter > self.update_target_after and terminal:
            self.update_target_model()


def main(args=None):
    if args is None:
        args = sys.argv
    stage_num = args[1] if len(args) > 1 else '1'
    max_training_episodes = args[2] if len(args) > 2 else '1000'
    rclpy.init(args=args)

    dqn_agent = DQNAgent(stage_num, max_training_episodes)
    rclpy.spin(dqn_agent)

    dqn_agent.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
