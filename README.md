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
