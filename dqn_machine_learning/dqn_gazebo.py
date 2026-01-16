#!/usr/bin/env python3

import os
import random
import subprocess
import sys
import time

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from std_srvs.srv import Empty

from turtlebot3_msgs.srv import Goal

ROS_DISTRO = os.environ.get('ROS_DISTRO')
if ROS_DISTRO == 'humble':
    from gazebo_msgs.srv import DeleteEntity
    from gazebo_msgs.srv import SpawnEntity
    from geometry_msgs.msg import Pose


class GazeboInterface(Node):

    def __init__(self, stage_num):
        super().__init__('gazebo_interface')

        self.entity_name = 'goal_box'
        self.entity_pose_x = 0.0
        self.entity_pose_y = 0.0

        # 보상 위치 좌표 설정하는곳
        self.goal_pose_candidates = [
            (-1.05,-4.0),
            (-1.05,-3.5),
            (-1.05,-3.0),
            (-1.05,-2.4),
            (-1.05,-1.8),
            (-2.0,-3.0),
            (-2.0,-2.5),
            (-1.5,-2.5)
            #(-3.0,-3.0),
            #(-2.6,-4.0)
        ]

        if ROS_DISTRO == 'humble':
            self.entity = None
            self.open_entity()
            self.delete_entity_client = self.create_client(DeleteEntity, 'delete_entity')
            self.spawn_entity_client = self.create_client(SpawnEntity, 'spawn_entity')
            self.reset_simulation_client = self.create_client(Empty, 'reset_simulation')

        self.callback_group = MutuallyExclusiveCallbackGroup()
        self.initialize_env_service = self.create_service(
            Goal,
            'initialize_env',
            self.initialize_env_callback,
            callback_group=self.callback_group
        )
        self.task_succeed_service = self.create_service(
            Goal,
            'task_succeed',
            self.task_succeed_callback,
            callback_group=self.callback_group
        )
        self.task_failed_service = self.create_service(
            Goal,
            'task_failed',
            self.task_failed_callback,
            callback_group=self.callback_group
        )
    # 가제보 모델 열기
    def open_entity(self):
        package_share = get_package_share_directory('turtlebot3_gazebo')
        model_path = os.path.join(
            package_share, 'models', 'turtlebot3_dqn_world', 'goal_box', 'model.sdf'
        )
        with open(model_path, 'r') as f:
            self.entity = f.read()
    # 보상 위치 좌표 생성 함수
    def generate_goal_pose(self):
        self.entity_pose_x, self.entity_pose_y = random.choice(self.goal_pose_candidates)

    # 보상 위치에 모델 스폰하는 함수
    def spawn_entity(self):
        if ROS_DISTRO == 'humble':
            entity_pose = Pose()
            entity_pose.position.x = self.entity_pose_x
            entity_pose.position.y = self.entity_pose_y

            spawn_req = SpawnEntity.Request()
            spawn_req.name = self.entity_name
            spawn_req.xml = self.entity
            spawn_req.initial_pose = entity_pose

            while not self.spawn_entity_client.wait_for_service(timeout_sec=1.0):
                pass
            future = self.spawn_entity_client.call_async(spawn_req)
            rclpy.spin_until_future_complete(self, future)
        else:
            service_name = '/world/dqn/create'
            package_share = get_package_share_directory('turtlebot3_gazebo')
            model_path = os.path.join(
                package_share, 'models', 'turtlebot3_dqn_world', 'goal_box', 'model.sdf'
            )
            req = (
                f'sdf_filename: "{model_path}", '
                f'name: "{self.entity_name}", '
                f'pose: {{ position: {{ '
                f'x: {self.entity_pose_x}, '
                f'y: {self.entity_pose_y}, '
                f'z: 0.0 }} }}'
            )
            cmd = [
                'gz', 'service',
                '-s', service_name,
                '--reqtype', 'gz.msgs.EntityFactory',
                '--reptype', 'gz.msgs.Boolean',
                '--timeout', '1000',
                '--req', req
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL)
    
    # 보상 모델 삭제 함수
    def delete_entity(self):
        if ROS_DISTRO == 'humble':
            delete_req = DeleteEntity.Request()
            delete_req.name = self.entity_name

            while not self.delete_entity_client.wait_for_service(timeout_sec=1.0):
                pass
            future = self.delete_entity_client.call_async(delete_req)
            rclpy.spin_until_future_complete(self, future)
        else:
            service_name = '/world/dqn/remove'
            req = f'name: "{self.entity_name}", type: 2'
            cmd = [
                'gz', 'service',
                '-s', service_name,
                '--reqtype', 'gz.msgs.Entity',
                '--reptype', 'gz.msgs.Boolean',
                '--timeout', '1000',
                '--req', req
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL)
    # 시뮬레이션 상태 리셋 함수
    def reset_simulation(self):
        reset_req = Empty.Request()
        while not self.reset_simulation_client.wait_for_service(timeout_sec=1.0):
            pass
        self.reset_simulation_client.call_async(reset_req)

    # 보상 위치 재설정 콜백 함수
    def task_succeed_callback(self, request, response):
        self.delete_entity()
        time.sleep(0.1) # 여기서 딜레이를 고치면 시뮬레이션 속도 개선 가능함 (하지만 너무 짧게해도 오류남)
        self.generate_goal_pose()
        time.sleep(0.1)
        self.spawn_entity()
        response.pose_x = self.entity_pose_x
        response.pose_y = self.entity_pose_y
        response.success = True
        return response
    # 실패시 콜백 함수
    def task_failed_callback(self, request, response):
        self.delete_entity()
        time.sleep(0.1)
        if ROS_DISTRO == 'humble':
            self.reset_simulation()
        time.sleep(0.1)
        self.generate_goal_pose()
        time.sleep(0.1)
        self.spawn_entity()
        response.pose_x = self.entity_pose_x
        response.pose_y = self.entity_pose_y
        response.success = True
        return response
    # 환경 초기화 콜백 함수
    def initialize_env_callback(self, request, response):
        self.delete_entity()
        time.sleep(0.1)
        if ROS_DISTRO == 'humble':
            self.reset_simulation()
        time.sleep(0.1)
        self.generate_goal_pose()
        time.sleep(0.1)
        self.spawn_entity()
        response.pose_x = self.entity_pose_x
        response.pose_y = self.entity_pose_y
        response.success = True
        return response


def main(args=None):
    rclpy.init(args=sys.argv)
    gazebo_interface = GazeboInterface(1)
    try:
        while rclpy.ok():
            rclpy.spin_once(gazebo_interface, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        gazebo_interface.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
