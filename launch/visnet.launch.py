import launch
from launch import LaunchDescription
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration

from ament_index_python.packages import get_package_share_directory

from datetime import date, datetime
date_str = f"{date.today().strftime('%Y-%m-%d')}-{datetime.now().time().strftime('%I%M%S')}"

def generate_launch_description():
    return LaunchDescription([
        # SetEnvironmentVariable(name='GSCAM_CONFIG', value="v4l2src device=/dev/video0 ! image/jpeg,width=1600,height=1200,framerate=30/1 ! jpegdec ! videoconvert"),
        DeclareLaunchArgument(
            'camera_calibration_file',
            default_value='file://' + get_package_share_directory('visnet') + '/config/camera.yaml'),

        Node(
            package = 'qualisys_ros',
            executable = 'qualisys_node',
            namespace = "qualisys",
        ),

        Node(
            package='gscam',
            executable='gscam_node',
            namespace='camera_0',
            parameters=[
                {'gscam_config': 'v4l2src device=/dev/video0 ! image/jpeg,width=1600,height=1200,framerate=30/1 ! jpegdec ! videoconvert'},
                {'camera_info_url': 'package://visnet/config/camera_0.yaml'},
                {'frame_id': 'camera_0'},
                {'camera_name': 'camera_0'}
            ],
            remappings=[
                ('camera/image_raw', 'image'),
                ('camera/camera_info', 'camera_info'),
                ('camera/image_raw/compressed', 'image/compressed'),
                ('camera/image_raw/compressedDepth', 'image/compressedDepth'),
                ('camera/image_raw/theora', 'image/theora'),
            ]
        ),

        Node(
            package='gscam',
            executable='gscam_node',
            namespace='camera_1',
            parameters=[
                {'gscam_config': 'v4l2src device=/dev/video2 ! image/jpeg,width=1600,height=1200,framerate=30/1 ! jpegdec ! videoconvert'},
                {'camera_info_url': 'package://visnet/config/camera_1.yaml'},
                {'frame_id': 'camera_1'},
                {'camera_name': 'camera_1'}
            ],
            remappings=[
                ('camera/image_raw', 'image'),
                ('camera/camera_info', 'camera_info'),
                ('camera/image_raw/compressed', 'image/compressed'),
                ('camera/image_raw/compressedDepth', 'image/compressedDepth'),
                ('camera/image_raw/theora', 'image/theora'),
            ]
        ),

        Node(
            package='gscam',
            executable='gscam_node',
            namespace='camera_3',
            parameters=[
                {'gscam_config': 'v4l2src device=/dev/video4 ! image/jpeg,width=1600,height=1200,framerate=30/1 ! jpegdec ! videoconvert'},
                {'camera_info_url': 'package://visnet/config/camera_3.yaml'},
                {'frame_id': 'camera_3'},
                {'camera_name': 'camera_3'}
            ],
            remappings=[
                ('camera/image_raw', 'image'),
                ('camera/camera_info', 'camera_info'),
                ('camera/image_raw/compressed', 'image/compressed'),
                ('camera/image_raw/compressedDepth', 'image/compressedDepth'),
                ('camera/image_raw/theora', 'image/theora'),
            ]
        ),

        # Node(
        #     package='image_proc',
        #     executable='image_proc',
        #     namespace='camera_0',
        #     output='screen',
        # ),

        # Node(
        #     package='image_proc',
        #     executable='image_proc',
        #     namespace='camera_1',
        #     output='screen',
        # ),

        # Node(
        #     package='image_proc',
        #     executable='image_proc',
        #     namespace='camera_3',
        #     output='screen',
        # ),


        Node(
           package='rviz2',
           executable='rviz2',
           arguments=['-d', get_package_share_directory('visnet') + '/config/camera_view.rviz']
        ),
        
        # launch.actions.ExecuteProcess(
        #     cmd=['ros2', 'bag', 'record',
        #         '-o', f'/home/zpyang/rosbags/multicam-{date_str}',
        #         '/camera_0/camera/image_raw/compressed',
        #         '/camera_1/camera/image_raw/compressed',
        #         '/camera_3/camera/image_raw/compressed',
        #         '/qualisys/camera_0/pose',
        #         '/qualisys/camera_1/pose',
        #         '/qualisys/camera_3/pose',
        #         '/qualisys/hb1/pose',
        #         ],
        #     output='screen'
        # )
    ])
