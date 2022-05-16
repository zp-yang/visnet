#!/usr/bin/env python3
import asyncio
import math
import xml.etree.cElementTree as ET
from threading import Thread
import qtm
import numpy as np

import rospy
from geometry_msgs.msg import PoseStamped

class QtmWrapper(Thread):
    def __init__(self, body_names, realtime=False, hz=10):
        Thread.__init__(self)

        self.realtime = realtime
        self.body_names = body_names

        self.connection = None
        self.qtm_6DoF_labels = []
        self._stay_open = True

        rospy.logdebug("creating publishers from input body names...")
        self.pose_pubs = {name: rospy.Publisher(f"/{name}/pose", PoseStamped, queue_size=10) for name in body_names}
        rospy.logdebug(f"publishers are created, {len(self.pose_pubs)} in total")
        self.last_send = rospy.get_rostime()
        self.dt_min = 1/hz # reducing send_extpose rate to 5HZ

        self.start()

    def close(self):
        self._stay_open = False
        self.join()

    def run(self):
        asyncio.run(self._life_cycle())

    async def _life_cycle(self):
        await self._connect()
        while(self._stay_open):
            await asyncio.sleep(1)
        await self._close()

    async def _connect(self):
        
        host = "192.168.123.202"
        rospy.loginfo('Connecting to QTM on %s', host)
        self.connection = await qtm.connect(host=host, version="1.20") # version 1.21 has weird 6DOF labels, so using 1.20 here
        
        if self.connection is None:
            rospy.logerror("Failed to connect to QTM!!!")
            return
        
        async with qtm.TakeControl(self.connection, "password"):

            if self.realtime:
                rospy.loginfo("Starting to stream realtime data...")
                await self.connection.new()
            else:
                rospy.loginfo("Starting to stream recorded data...")
                QTM_FILE = "C:/Users/QTM/Documents/05-03-22_passive_high_pression/Data/cam_pos_test.qtm"
                await self.connection.load(QTM_FILE)
                await self.connection.start(rtfromfile=True)
        rospy.loginfo("QTM connected!")

        params = await self.connection.get_parameters(parameters=['6d'])
        xml = ET.fromstring(params)
        self.qtm_6DoF_labels = [label.text for label in xml.iter('Name')]

        await self.connection.stream_frames(
            components=['6D'],
            on_packet=self._on_packet)
    
    # callback function whenever an RT packet is received
    def _on_packet(self, packet):
        now = rospy.get_rostime()
        dt = (now - self.last_send).to_sec()
        if dt < self.dt_min:
            return
        self.last_send = rospy.get_rostime()
        # print('Hz: ', 1.0/dt)
        
        header, bodies = packet.get_6d()

        if bodies is None:
            return

        intersect = set(self.qtm_6DoF_labels).intersection(self.body_names) 
        if len(intersect) < len(self.body_names) :
            rospy.logerror('Missing rigid bodies')
            rospy.logerror(f"Objects to track: {self.bodynames}")
            rospy.logerror(f"Objects in QTM: {str(self.qtm_6DoF_labels)}")
            rospy.logerror(f'Intersection: {str(intersect)}')
            return
        else:
            for body_name in self.body_names:
                index = self.qtm_6DoF_labels.index(body_name)
                temp_pose = bodies[index]
                x = temp_pose[0][0] / 1000
                y = temp_pose[0][1] / 1000
                z = temp_pose[0][2] / 1000

                r = temp_pose[1].matrix
                rot = [
                    [r[0], r[3], r[6]],
                    [r[1], r[4], r[7]],
                    [r[2], r[5], r[8]],
                ]

                if self.pose_pubs[body_name]:
                    # Make sure we got a position
                    if math.isnan(x):
                        rospy.logerror("======= LOST RB TRACKING : %s", body_name)
                        continue
                    pose = PoseStamped()
                    pose.header.stamp = rospy.get_rostime()
                    pose.header.frame_id = body_name
                    pose.pose.position.x = x
                    pose.pose.position.y = y
                    pose.pose.position.z = z
                    pose.pose.orientation.w = 1
                    pose.pose.orientation.x = 0
                    pose.pose.orientation.y = 0
                    pose.pose.orientation.z = 0
                    
                    self.pose_pubs[body_name].publish(pose)
                    rospy.loginfo(f"{body_name} pose published: x {x}, y {y}, z {z}")

    async def _close(self):
        await self.connection.stream_frames_stop()
        self.connection.disconnect()

def _sqrt(a):
    """
    There might be rounding errors making 'a' slightly negative.
    Make sure we don't throw an exception.
    """
    if a < 0.0:
        return 0.0
    return math.sqrt(a)


def main():
    body_names = [
        "camera_0",
        "camera_1",
        "camera_2",
        "camera_3",
        ]
    rospy.init_node("qualisys_node")
    rospy.loginfo("finished shitfuck")
    qtm = QtmWrapper(body_names=body_names, realtime=False)
    
    rospy.spin()

if __name__=="__main__":
    main()

