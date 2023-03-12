"""
Python implementation of Offboard Control

"""


import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import VehicleControlMode
from px4_msgs.msg import VehicleOdometry

import os
import numpy as np

qos_profile_pub = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10
)
qos_profile_sub = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10
)

def wait_for_message(node : Node, msg_type, topic, time_out=10):
    import time
    class _wfm(object):
        def __init__(self) -> None:
            self.time_out = time_out
            self.msg = None
        
        def cb(self, msg):
            self.msg = msg
    elapsed = 0
    wfm = _wfm()
    subscription = node.create_subscription(msg_type, topic, wfm.cb, qos_profile=qos_profile_sub)
    # rate = node.create_rate(10)
    while rclpy.ok():
        if wfm.msg != None : return wfm.msg
        node.get_logger().info(f'waiting for {topic} ...')
        rclpy.spin_once(node)
        time.sleep(0.1)
        elapsed += 0.1
        if elapsed >= wfm.time_out:
            node.get_logger().warn(f'time out waiting for {topic}...')
            return None
    subscription.destroy()

class OffboardControl(Node):

    def __init__(self):
        super().__init__('OffboardControl')

        PX4_NS = os.getenv("PX4_MICRODDS_NS")
        fmu = f"{PX4_NS}/fmu"

        
        
        self.offboard_control_mode_publisher_ = self.create_publisher(OffboardControlMode, f"{fmu}/in/offboard_control_mode", qos_profile_pub)
        self.trajectory_setpoint_publisher_ = self.create_publisher(TrajectorySetpoint, f"{fmu}/in/trajectory_setpoint", qos_profile_pub)
        self.vehicle_command_publisher_ = self.create_publisher(VehicleCommand, f"{fmu}/in/vehicle_command", qos_profile_pub)

        self.vehicle_odom_sub = self.create_subscription(VehicleOdometry, f'{fmu}/out/vehicle_odometry', self.vehicle_odom_callback, qos_profile_sub)

        self.offboard_setpoint_counter_ = 0

        timer_period = 0.1  # 100 milliseconds
        self.timer_ = self.create_timer(timer_period, self.timer_callback)

        self.start_odom = wait_for_message(self, VehicleOdometry, f'{fmu}/out/vehicle_odometry')

        self.cur_pos = None
        self.cur_vel = None

    def vehicle_odom_callback(self, msg):
        self.cur_pos = msg.position
        self.cur_vel = msg.velocity

    def timer_callback(self):
        if (self.offboard_setpoint_counter_ == 10):
            # Change to Offboard mode after 10 setpoints
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1., 6.)
            # Arm the vehicle
            self.arm()
            
        # Offboard_control_mode needs to be paired with trajectory_setpoint
        self.publish_offboard_control_mode()
        self.publish_trajectory_setpoint()

        # stop the counter after reaching 11
        if (self.offboard_setpoint_counter_ < 11):
            self.offboard_setpoint_counter_ += 1

    # Arm the vehicle
    def arm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        self.get_logger().info("Arm command send")

    # Disarm the vehicle
    def disarm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)
        self.get_logger().info("Disarm command send")

    '''
	Publish the offboard control mode.
	For this example, only position and altitude controls are active.
    '''

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(Clock().now().nanoseconds / 1000) # time in microseconds
        self.offboard_control_mode_publisher_.publish(msg)

    '''
	Publish a trajectory setpoint
	For this example, it sends a trajectory setpoint to make the
	vehicle hover at 5 meters with a yaw angle of 180 degrees.
    '''

    def publish_trajectory_setpoint(self):
        if self.start_odom is not None:
            msg = TrajectorySetpoint()
            msg.timestamp = int(Clock().now().nanoseconds / 1000)
            msg.position = np.array([self.start_odom.position[0], self.start_odom.position[1], -5.0]).astype(np.float32) 
            msg.yaw = 3.14  # [-PI:PI]
            msg.timestamp = int(Clock().now().nanoseconds / 1000) # time in microseconds
            self.trajectory_setpoint_publisher_.publish(msg)

    '''
    Publish vehicle commands
        command   Command code (matches VehicleCommand and MAVLink MAV_CMD codes)
        param1    Command parameter 1 as defined by MAVLink uint16 VEHICLE_CMD enum
        param2    Command parameter 2 as defined by MAVLink uint16 VEHICLE_CMD enum
    '''
    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.param1 = param1
        msg.param2 = param2
        msg.command = command  # command ID
        msg.target_system = 0  # system which should execute the command
        msg.target_component = 1  # component which should execute the command, 0 for all components
        msg.source_system = 1  # system sending the command
        msg.source_component = 1  # component sending the command
        msg.from_external = True
        msg.timestamp = int(Clock().now().nanoseconds / 1000) # time in microseconds
        self.vehicle_command_publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    print("Starting offboard control node...\n")
    offboard_control = OffboardControl()
    rclpy.spin(offboard_control)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
