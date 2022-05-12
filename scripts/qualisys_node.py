#!/usr/bin/env python3
import time
import asyncio
import math
import xml.etree.cElementTree as ET
from threading import Thread
import pkg_resources
import qtm
QTM_FILE = pkg_resources.resource_filename("qtm", "data/Demo.qtm")
import numpy as np

import logging
logger = logging.getLogger("qualisys_node")
logging.getLogger("qtm").setLevel(logging.WARN)

import rospy
from geometry_msgs.msg import PoseStamped

class QtmWrapper(Thread):
    def __init__(self, body_names, realtime=False):
        Thread.__init__(self)

        self.realtime = realtime
        self.body_names = body_names
        self.on_pose = {name: None for name in body_names}
        self.connection = None
        self.qtm_6DoF_labels = []
        self._stay_open = True

        self.last_send = time.time()
        self.dt_min = 0.2 # reducing send_extpose rate to 5HZ

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
        if self.realtime:
            host = "192.168.1.2"
            logger.info('Connecting to QTM on %s', host)
            self.connection = await qtm.connect(host=host, version="1.20") # version 1.21 has weird 6DOF labels, so using 1.20 here

            params = await self.connection.get_parameters(parameters=['6d'])
            xml = ET.fromstring(params)
            self.qtm_6DoF_labels = [label.text for label in xml.iter('Name')]

            await self.connection.stream_frames(
                components=['6D'],
                on_packet=self._on_packet)
        else:
            logger.info("Streaming recorded data...")

    def _on_packet(self, packet):
        now = time.time()
        dt = now - self.last_send
        if dt < self.dt_min:
            return
        self.last_send = time.time()
        # print('Hz: ', 1.0/dt)
        
        header, bodies = packet.get_6d()

        if bodies is None:
            return

        intersect = set(self.qtm_6DoF_labels).intersection(self.body_names) 
        if len(intersect) < len(self.body_names) :
            logger.error('Missing rigid bodies')
            logger.error('In QTM: %s', str(self.qtm_6DoF_labels))
            logger.error('Intersection: %s', str(intersect))
            return            
        else:
            for body_name in self.body_names:
                index = self.qtm_6DoF_labels.index(body_name)
                temp_cf_pos = bodies[index]
                x = temp_cf_pos[0][0] / 1000
                y = temp_cf_pos[0][1] / 1000
                z = temp_cf_pos[0][2] / 1000

                r = temp_cf_pos[1].matrix
                rot = [
                    [r[0], r[3], r[6]],
                    [r[1], r[4], r[7]],
                    [r[2], r[5], r[8]],
                ]

                if self.on_pose[body_name]:
                    # Make sure we got a position
                    if math.isnan(x):
                        logger.error("======= LOST RB TRACKING : %s", body_name)
                        continue

                    self.on_pose[body_name]([x, y, z, rot])

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
        "cam0",
        "cam1",
        "cam2",
        "cam3",
        "cf0",
    ]
    qtm = QtmWrapper(body_names=body_names, realtime=True)

if __name__=="__main__":
    main()

