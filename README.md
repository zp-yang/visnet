# visnet
Network of cameras to identify and localize UAVs in urban environment.

Put this project(folder) under your ros2 workspace `src` folder

`ros2_ws/src/visnet/`

under `ros2_ws` do

`colcon build --packages-select visnet`

To source the project

`. install/setup.bash`

To launch simulator (classic gazebo)

Install classic gazebo first`sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros`

`ros2 launch visnet gz_sim.launch.py`


****
#### Running with real cameras

Depends on `gscam` package for ros as the camera driver.

To install `gscam`,
`sudo apt install ros-humble-gscam`

**gscam dependencies**
`sudo apt-get install gstreamer1.0-tools libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev`

**gstreamer pipline depends on v4l2**
`sudo apt install v4l-utils`

use `v4l2-ctl --list-formats-ext` to check available format for the gstreamer pipeline. Default setting in this launch script for Arducam is MJPEG 1600x1200 at 30 fps.
the pipeline looks like 
`v4l2src device=/dev/video0 ! image/jpeg,width=1600,height=1200,framerate=30/1 ! jpegdec ! videoconvert`

To launch real cameras after building and sourcing the package
`ros2 launch visnet gscam.launch.py`

For more information about how to use `gscam` with ros2 please check 
https://github.com/ros-drivers/gscam/tree/ros2

***
#### Using ros2 bag

To record a bag file for the session, uncomment the last part of `gscam.launch.py` then build and run the launch script.

You can specifiy the output location of the bag in the launch script or it will be default to the current directory of the terminal.

To run ros2 bag, locate the path of the recorded bag and modify `bag.launch.py` to match the file path. 
Then build and run `ros2 launch visnet bag.launch.py`.

By default `rviz2` and `rqt` will launch but there will be no image in `rviz2` beceause current `rviz2` cannot visualize compressed images. Thus we will use `rqt` to visualize compressed image topics.
