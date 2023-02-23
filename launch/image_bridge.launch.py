# Copyright 2019 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():

    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    pkg_visnet = get_package_share_directory('visnet')

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={
            'gz_args': f'-r {pkg_visnet}/worlds/purdue.sdf'
        }.items(),
    )

    # RQt
    # rqt = Node(
    #     package='rqt_image_view',
    #     executable='rqt_image_view',
    #     arguments=[LaunchConfiguration('image_topic')],
    #     condition=IfCondition(LaunchConfiguration('rqt'))
    # )

    # Image Bridge
    bridge = Node(
        package='ros_gz_image',
        executable='image_bridge',
        arguments=['camera_0','camera_1'],
        output='screen'
    )

    return LaunchDescription([
        gz_sim,
        # DeclareLaunchArgument('rqt', default_value='true',
        #                       description='Open RQt.'),
        DeclareLaunchArgument('image_topic', default_value='/camera_0',
                              description='Topic to start viewing in RQt.'),
        bridge,
        # rqt
    ])