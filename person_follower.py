import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from ultralytics import YOLO
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from cv_bridge import CvBridge
qos = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.BEST_EFFORT
)

class PersonFollower(Node):
    def __init__(self):
        super().__init__('person_follower')


        self.model = YOLO('yolov8n.pt')
        self.bridge = CvBridge()

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.img_pub = self.create_publisher(CompressedImage, '/detected_image/compressed', 10)
        self.img_sub = self.create_subscription(CompressedImage,'/image_raw/compressed',self.image_callback,qos)

        self.frame = None
        self.get_logger().info('사람 추종 시작')
        self.last_process_time = 0.0
        self.process_interval = 0.2

    def image_callback(self, msg):
        current_time = self.get_clock().now().nanoseconds / 1e9
        if current_time - self.last_process_time < self.process_interval:
            return

        try:
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            frame = cv2.resize(frame, (320, 240))
            if frame is None:
                return

            annotated_frame = self.process_frame(frame)
            annotated_msg = self.bridge.cv2_to_compressed_imgmsg(annotated_frame)
            self.img_pub.publish(annotated_msg)

            self.last_process_time = current_time
        except Exception as e:
            self.get_logger().error(str(e))

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        frame_center_x = w / 2

        results = self.model(frame, conf=0.6, imgsz=320, verbose=False)
        twist = Twist()
        person_detected = False
        annotated_frame = frame.copy()

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls != 0:   
                    continue

                person_detected = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_center_x = (x1 + x2) / 2
                error = person_center_x - frame_center_x
                area = (x2 - x1) * (y2 - y1)

                if area < 50000:
                    twist.linear.x = 0.2
                else:
                    twist.linear.x = 0.0

                twist.angular.z = max(min(-error / 200.0, 0.6), -0.6)

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame,'person',(x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),2)
                break

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