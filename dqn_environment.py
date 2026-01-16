#!/usr/bin/env python3

import math
import os

from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
import numpy
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

from turtlebot3_msgs.srv import Dqn, Goal

ROS_DISTRO = os.environ.get('ROS_DISTRO')
COLLISION_DIST = 0.15


class RLEnvironment(Node):

    def __init__(self):
        super().__init__('rl_environment')

        self.goal_pose_x = 0.0
        self.goal_pose_y = 0.0
        self.robot_pose_x = 0.0
        self.robot_pose_y = 0.0

        self.action_size = 5
        self.max_step = 800

        self.done = False
        self.fail = False
        self.succeed = False
        self.state_size = 14

        self.goal_angle = 0.0
        self.goal_distance = 1.0
        self.init_goal_distance = 0.5
        self.prev_goal_distance = 0.5

        self.scan_ranges = []
        self.front_ranges = []
        self.front_angles = []

        self.min_obstacle_distance = 10.0

        self.local_step = 0
        self.stop_cmd_vel_timer = None

        self.angular_vel = [1.0, 0.5, 0.0, -0.5, -1.0]

        qos = QoSProfile(depth=10)

        if ROS_DISTRO == 'humble':
            self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', qos)
        else:
            self.cmd_vel_pub = self.create_publisher(TwistStamped, 'cmd_vel', qos)

        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_sub_callback, qos)
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_sub_callback, qos_profile_sensor_data)

        self.clients_callback_group = MutuallyExclusiveCallbackGroup()

        self.task_succeed_client = self.create_client(Goal, 'task_succeed', callback_group=self.clients_callback_group)
        self.task_failed_client = self.create_client(Goal, 'task_failed', callback_group=self.clients_callback_group)
        self.initialize_environment_client = self.create_client(
            Goal, 'initialize_env', callback_group=self.clients_callback_group
        )

        self.rl_agent_interface_service = self.create_service(
            Dqn, 'rl_agent_interface', self.rl_agent_interface_callback
        )
        self.make_environment_service = self.create_service(
            Empty, 'make_environment', self.make_environment_callback
        )
        self.reset_environment_service = self.create_service(
            Dqn, 'reset_environment', self.reset_environment_callback
        )

    def make_environment_callback(self, request, response):
        while not self.initialize_environment_client.wait_for_service(timeout_sec=1.0):
            pass
        future = self.initialize_environment_client.call_async(Goal.Request())
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        self.goal_pose_x = result.pose_x
        self.goal_pose_y = result.pose_y
        return response

    def reset_environment_callback(self, request, response):
        state = self.calculate_state()
        self.init_goal_distance = state[0]
        self.prev_goal_distance = self.init_goal_distance
        response.state = state
        return response

    def call_task_succeed(self):
        while not self.task_succeed_client.wait_for_service(timeout_sec=1.0):
            pass
        future = self.task_succeed_client.call_async(Goal.Request())
        rclpy.spin_until_future_complete(self, future)
        res = future.result()
        self.goal_pose_x = res.pose_x
        self.goal_pose_y = res.pose_y

    def call_task_failed(self):
        while not self.task_failed_client.wait_for_service(timeout_sec=1.0):
            pass
        future = self.task_failed_client.call_async(Goal.Request())
        rclpy.spin_until_future_complete(self, future)
        res = future.result()
        self.goal_pose_x = res.pose_x
        self.goal_pose_y = res.pose_y

    def scan_sub_callback(self, scan):
        self.scan_ranges = []
        self.front_ranges = []
        self.left_ranges = []
        self.front_center_ranges = []
        self.right_ranges = []

        for i, distance in enumerate(scan.ranges):
            angle = scan.angle_min + i * scan.angle_increment

            if distance == float('Inf'):
                distance = 3.5
            elif numpy.isnan(distance):
                distance = 0.0

            self.scan_ranges.append(distance)

            if -math.pi / 2 <= angle <= math.pi / 2:
                self.front_ranges.append(distance)

                if math.pi / 6 < angle <= math.pi / 2:
                    self.left_ranges.append(distance)
                elif -math.pi / 6 <= angle <= math.pi / 6:
                    self.front_center_ranges.append(distance)
                elif -math.pi / 2 <= angle < -math.pi / 6:
                    self.right_ranges.append(distance)

        if self.front_ranges:
            self.min_obstacle_distance = min(self.front_ranges)
        else:
            self.min_obstacle_distance = 3.5

    def odom_sub_callback(self, msg):
        self.robot_pose_x = msg.pose.pose.position.x
        self.robot_pose_y = msg.pose.pose.position.y
        _, _, self.robot_pose_theta = self.euler_from_quaternion(msg.pose.pose.orientation)

        self.goal_distance = math.sqrt(
            (self.goal_pose_x - self.robot_pose_x) ** 2 +
            (self.goal_pose_y - self.robot_pose_y) ** 2
        )

        path_theta = math.atan2(
            self.goal_pose_y - self.robot_pose_y,
            self.goal_pose_x - self.robot_pose_x
        )

        self.goal_angle = path_theta - self.robot_pose_theta
        if self.goal_angle > math.pi:
            self.goal_angle -= 2 * math.pi
        elif self.goal_angle < -math.pi:
            self.goal_angle += 2 * math.pi

    def calculate_state(self):
        left = min(self.left_ranges) if self.left_ranges else 3.5
        center = min(self.front_center_ranges) if self.front_center_ranges else 3.5
        right = min(self.right_ranges) if self.right_ranges else 3.5

        state = [
            float(self.goal_distance),
            float(self.goal_angle),
            left,
            center,
            right
        ]

        self.local_step += 1
        if self.local_step < 10:
            return state

        if self.goal_distance < 0.25:
            self.succeed = True
            self.done = True
            self.local_step = 0
            self.call_task_succeed()
            if self.succeed:
                self.cmd_vel_pub.publish(Twist())
            return state

        if self.min_obstacle_distance < 0.25:
            self.fail = True
            self.done = True
            self.local_step = 0
            self.call_task_failed()
            return state

        if self.local_step >= self.max_step:
            self.fail = True
            self.done = True
            self.local_step = 0
            self.call_task_failed()
            return state

        return state

    def compute_weighted_obstacle_reward(self):
        if not self.front_ranges:
            return 0.0

        front_ranges = numpy.array(self.front_ranges)
        mask = front_ranges < 0.5
        if not numpy.any(mask):
            return 0.0

        dists = numpy.clip(front_ranges[mask] - 0.25, 1e-2, 3.5)
        return -numpy.mean(numpy.exp(-3.0 * dists))

    def calculate_reward(self):
        distance_reward = self.prev_goal_distance - self.goal_distance
        self.prev_goal_distance = self.goal_distance

        yaw_reward = 1.0 - abs(self.goal_angle) / math.pi
        obstacle_reward = self.compute_weighted_obstacle_reward()

        collision_penalty = 0.0
        if self.min_obstacle_distance < 0.5:
            collision_penalty = -5.0 * math.exp(-4.0 * self.min_obstacle_distance)

        reward = (2.0 * distance_reward +1.0 * yaw_reward +obstacle_reward +collision_penalty)

        if self.succeed:
            reward = 100.0
        elif self.fail:
            reward = -200.0

        return reward

    def rl_agent_interface_callback(self, request, response):
        action = request.action

        #right_front = min(self.front_ranges[len(self.front_ranges)//2:])
        #left_front  = min(self.front_ranges[:len(self.front_ranges)//2])

        #SAFE_DIST = 0.25


        #if right_front < SAFE_DIST and action in [3, 4]:
            #action = 1


        #elif left_front < SAFE_DIST and action in [0, 1]:
            #action = 3




        if ROS_DISTRO == 'humble':
            msg = Twist()
            msg.linear.x = 0.2
            msg.angular.z = self.angular_vel[action]
        else:
            msg = TwistStamped()
            msg.twist.linear.x = 0.2
            msg.twist.angular.z = self.angular_vel[action]

        self.cmd_vel_pub.publish(msg)

        if self.stop_cmd_vel_timer:
            self.destroy_timer(self.stop_cmd_vel_timer)
        self.stop_cmd_vel_timer = self.create_timer(0.2, self.timer_callback)

        response.state = self.calculate_state()
        response.reward = self.calculate_reward()
        response.done = self.done

        if self.done:
            self.done = False
            self.succeed = False
            self.fail = False

        return response

    def timer_callback(self):
        if ROS_DISTRO == 'humble':
            self.cmd_vel_pub.publish(Twist())
        else:
            self.cmd_vel_pub.publish(TwistStamped())
        self.destroy_timer(self.stop_cmd_vel_timer)

    def euler_from_quaternion(self, quat):
        x, y, z, w = quat.x, quat.y, quat.z, quat.w

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = numpy.arctan2(siny_cosp, cosy_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = numpy.arcsin(sinp)

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = numpy.arctan2(sinr_cosp, cosr_cosp)

        return roll, pitch, yaw


def main(args=None):
    rclpy.init(args=args)
    node = RLEnvironment()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
