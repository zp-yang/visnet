#!/usr/bin/
gz service -s /world/purdue/set_pose \
--reqtype gz.msgs.Pose \
--reptype gz.msgs.Boolean \
--timeout 300 \
-r \
"name: 'camera_0', \
position: {x: 0, y: 15, z: 1}"