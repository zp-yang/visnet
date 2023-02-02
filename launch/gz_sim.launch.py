# Author: Addison Sears-Collins
# Date: September 23, 2021
# Description: Load a world file into Gazebo.
# https://automaticaddison.com

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node, SetRemap
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # Set the path to the Gazebo ROS package
    # pkg_gazebo_ros = FindPackageShare(package="gazebo_ros").find("gazebo_ros")
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    # Set the path to this package.
    # pkg_share = FindPackageShare(package="visnet").find("visnet")
    pkg_share = get_package_share_directory("visnet")

    # Set the path to the world file
    world_file_name = "visnet_gz_nocam.world"
    world_path = os.path.join(pkg_share, "worlds", world_file_name)

    # Set the path to the python scripts
    script_dir = os.path.join(pkg_share, "scripts") 

    # Set the path to the SDF model files.
    gazebo_models_path = os.path.join(pkg_share, "models")
    os.environ["GAZEBO_MODEL_PATH"] = gazebo_models_path

    ########### Gazebo Launch Command ##############
    # Launch configuration variables specific to simulation
    headless = LaunchConfiguration("headless")
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_simulator = LaunchConfiguration("use_simulator")
    world = LaunchConfiguration("world")

    declare_simulator_cmd = DeclareLaunchArgument(
        name="headless",
        default_value="False",
        description="Whether to execute gzclient",
    )

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        name="use_sim_time",
        default_value="true",
        description="Use simulation (Gazebo) clock if true",
    )

    declare_use_simulator_cmd = DeclareLaunchArgument(
        name="use_simulator",
        default_value="True",
        description="Whether to start the simulator",
    )

    declare_world_cmd = DeclareLaunchArgument(
        name="world",
        default_value=world_path,
        description="Full path to the world model file to load",
    )

    declare_pause_cmd = DeclareLaunchArgument(
        name="pause",
        default_value="true",
        description="Start the gzserver paused or unpaused",
    )

    # Specify the actions

    # Start Gazebo server
    start_gazebo_server_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, "launch", "gzserver.launch.py")
        ),
        condition=IfCondition(use_simulator),
        launch_arguments={"world": world, "pause": 'false'}.items(),
    )

    # Start Gazebo client
    start_gazebo_client_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, "launch", "gzclient.launch.py")
        ),
        condition=IfCondition(PythonExpression([use_simulator, " and not ", headless])),
    )

    # Create the launch description and populate
    ld = LaunchDescription()

    # Declare the launch options
    ld.add_action(declare_simulator_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_use_simulator_cmd)
    ld.add_action(declare_world_cmd)
    ld.add_action(declare_pause_cmd)

    # Add any actions
    ld.add_action(start_gazebo_server_cmd)
    ld.add_action(start_gazebo_client_cmd)

    start_rviz = Node(
           package='rviz2',
           executable='rviz2',
           arguments=['-d', pkg_share + '/config/sim_camera_view.rviz']
        )
    ld.add_action(start_rviz)

    # move_entity = ExecuteProcess(
    #     cmd=['python3', script_dir+'/set_entity_state.py'],
    #     output='screen'
    # )
    # ld.add_action(move_entity)

    # get entity state and publish PoseStamped message for ros2
    get_entity = ExecuteProcess(
        cmd=['python3', script_dir+'/get_entity_state.py'],
        output='screen'
    )
    ld.add_action(get_entity)

    return ld
