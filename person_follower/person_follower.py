import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from ultralytics import YOLO
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from cv_bridge import CvBridge

# qos 설정 - 이미지 토픽은 Best Effort로 설정
qos = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.BEST_EFFORT
)

class PersonFollower(Node):
    def __init__(self):
        super().__init__('person_follower')
        
        # YOLO 모델 로드 해오기
        self.model = YOLO('yolov8n.pt')
        self.bridge = CvBridge()

        # 로봇 제어 명령 퍼블리셔들
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.img_pub = self.create_publisher(CompressedImage, '/detected_image/compressed', 10)
        self.img_sub = self.create_subscription(CompressedImage,'/image_raw/compressed',self.image_callback,qos)

        self.frame = None
        self.get_logger().info('사람 추종 시작')

        self.last_process_time = 0.0
        self.process_interval = 0.2 # 0.2초마다 프레임 처리하는 인터벌을 설정했음

    def image_callback(self, msg):
        # 현재 시간 가져오기
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # 일정 주기(process_interval)보다 짧으면 리턴(초기 모델에 CPU 과부하를 막기위해 설정했고 현재 결과물에선 사용하지않음)
        if current_time - self.last_process_time < self.process_interval:
            return

        try:
            # ROS 메시지를 OpenCV 이미지로 변환하기
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            frame = cv2.resize(frame, (320, 240))
            
            if frame is None:
                return

            # 프레임 처리(사람 탐지 및 제어 명령 생성)
            annotated_frame = self.process_frame(frame)
            # 탐지 결과 이미지를 다시 ROS 메시지로 변환하여 퍼블리시함
            annotated_msg = self.bridge.cv2_to_compressed_imgmsg(annotated_frame)
            self.img_pub.publish(annotated_msg)
            
            ## 마지막 처리 시간 갱신
            self.last_process_time = current_time
        except Exception as e:
            self.get_logger().error(str(e))

    def process_frame(self, frame):
        # 프레임 크기 및 중심 좌표 계산
        h, w = frame.shape[:2]
        frame_center_x = w / 2
        
        # 여기서 YOLO 모델로 사람 탐지 수행함
        results = self.model(frame, conf=0.6, imgsz=320, verbose=False)
        twist = Twist() # 로봇제어
        person_detected = False
        annotated_frame = frame.copy()

        #탐지 결과 확인
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls != 0:  # 클래스 0이 사람이고 다른 객체는 무시하게 함   
                    continue

                person_detected = True

                # 탐지된 사람의 좌표
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_center_x = (x1 + x2) / 2
                error = person_center_x - frame_center_x # 사람 위치와 화면 중심 차이
                area = (x2 - x1) * (y2 - y1) # 사람 박스 크기(거리추정에 사용)

                # 사람이 멀리가면 박스크기가 작아지므로 추종명령, 가까워지면 멈춤
                if area < 50000:
                    twist.linear.x = 0.2
                else:
                    twist.linear.x = 0.0
                
                # 사람 위치에 따라 회전 (좌우로 회전하면서 보정) error값이 양수면 오른쪽에 사람이 있는것이며 0.6은 최대 회전 속도
                twist.angular.z = max(min(-error / 200.0, 0.6), -0.6)
                
                # 탐지된 사람 박스 그리기
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame,'person',(x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),2)
                break

        # 사람이 아니면 정지 명령
        if not person_detected:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        self.cmd_pub.publish(twist)

        return annotated_frame


def main(args=None):
    rclpy.init(args=args)
    node = PersonFollower()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()



if __name__ == '__main__':
    main()