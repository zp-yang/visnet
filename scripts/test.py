from os import kill
import numpy as np
import cv2
import os

print(cv2.__version__)


try:
    while True:
        cv2.imshow('1', np.random.randint(0,255, (1000, 1000, 3), np.uint8))
        cv2.imshow('2', np.random.randint(0,255, (1000, 1000, 3), np.uint8))
        k = cv2.waitKey(3) & 0xff
        print(k)
        if k == 27: break
except KeyboardInterrupt:
    cv2.destroyAllWindows()
    # kill(os.getpid,)