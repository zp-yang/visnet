#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from gazebo_msgs.srv import GetEntityState
from geometry_msgs.msg import PoseStamped

class GetTargetTraj(Node):
    def __init__(self, target_name):
        super().__init__(f"get_{target_name}_trajectory")
        client_cb_group = MutuallyExclusiveCallbackGroup()
        timer_cb_group  = MutuallyExclusiveCallbackGroup()

        self.client = self.create_client(GetEntityState, "get_entity_state", callback_group=client_cb_group)

        rate = self.create_rate(2.0)
        while not self.client.wait_for_service(timeout_sec=10.0):
            self.get_logger().info('gazebo get_entity_state service is not available, waiting...')
            rate.sleep()
        
        self.target_name = target_name
        timer_period = 0.01
        self.timer = self.create_timer(timer_period, self.timer_callback, callback_group=timer_cb_group)

        self.pose_pub = self.create_publisher(PoseStamped, f"/{target_name}/pose", 10)

        self.elapsed = 0
        self.time_last = self.get_clock().now()

        self.request = GetEntityState.Request()

    def timer_callback(self):
        self.elapsed += (self.get_clock().now()-self.time_last).nanoseconds / 1e9
        self.time_last = self.get_clock().now()

        self.request.name = self.target_name
        self.request.reference_frame = "world"

        self.result = self.client.call(self.request)
        pose = self.result.state.pose
        pose_msg = PoseStamped()
        pose_msg.pose = pose
        pose_msg.header.frame_id = "map"
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        self.pose_pub.publish(pose_msg)

def main(args=None):
        rclpy.init(args=args)

        executor = MultiThreadedExecutor()
        target_0 = GetTargetTraj("hb1")
        # target_1 = GetTargetTraj("drone_1")

        executor.add_node(target_0)
        # executor.add_node(target_1)
        try:
            executor.spin()
        except KeyboardInterrupt:
            print()

        target_0.destroy_node()
        # target_1.destroy_node()
        
        # rclpy.shutdown()
        
if __name__=='__main__':
    main()
        
