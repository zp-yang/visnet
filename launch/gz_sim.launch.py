import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.conditions import IfCondition
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    SetEnvironmentVariable(name='DISPLAY', value='0')
    pkg_ros_ign_gazebo = get_package_share_directory('ros_ign_gazebo')
    pkg_visnet = get_package_share_directory('visnet')

    ign_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_ign_gazebo, 'launch', 'ign_gazebo.launch.py')),
        launch_arguments={
            # 'ign_args': '-v 4 -r trial_1.world --gui-config /home/docker/.ignition/gazebo/6/gui.config'
            'ign_args': pkg_visnet + '/worlds/sim.world'
        }.items(),
    )

    bridge = Node(
        package='ros_ign_bridge',
        executable='parameter_bridge',
        arguments=[
            '/world/visnet/model/camera_0/link/link/sensor/camera/image@sensor_msgs/msg/Image@ignition.msgs.Image',
            '/world/visnet/model/camera_0/link/link/sensor/camera/camera_info@sensor_msgs/msg/CameraInfo@ignition.msg.CameraInfo',
        ],
        output='screen'
    )

    # rviz = Node(
    #     package='rviz2',
    #     executable='rviz2',
    #     arguments=['-d', os.path.join(pkg_auav_f22_gazebo, 'rviz', 'drone.rviz')],
    #     condition=IfCondition(LaunchConfiguration('rviz'))
    # )

    return LaunchDescription([
        ign_gazebo,
        DeclareLaunchArgument('rviz', default_value='true', description='Open RViz.'),
        bridge,
        # rviz,
    ])