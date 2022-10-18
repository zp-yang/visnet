import launch
from launch import LaunchDescription
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    return LaunchDescription([
        
        Node(
           package='rviz2',
           executable='rviz2',
           arguments=['-d', get_package_share_directory('visnet') + '/config/bag_view.rviz']
        ),
        launch.actions.ExecuteProcess(
            cmd=['ros2', 'bag', 'play',
                '/home/zpyang/rosbags/multicam-2022-10-17-051910',
                '-l'
                ],
            output='screen'
        ),

        launch.actions.ExecuteProcess(
            cmd=['rqt'],
            output='screen',
        )
    ])