import numpy as np
import cv2

print(cv2.__version__)
cv2.imshow('fuck', np.random.randint(0,255, (1000,1000,3)))
cv2.waitKey(0)
cv2.destroyAllWindows()