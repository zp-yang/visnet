# visnet
Network of cameras to identify and localize UAVs in urban environment.

Put this project(folder) under your ros2 workspace `src` folder

`ros2_ws/src/visnet/`

under `ros2_ws` do

`colcon build --packages-select visnet`

To source the project

`. install/setup.bash`

To launch simulator

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

