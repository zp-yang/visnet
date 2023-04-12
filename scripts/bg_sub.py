import cv2
import numpy as np


"""
Detects moving objects in the frame, and returns the bounding boxes
"""
subtractor_name = "MOG2"

if subtractor_name == 'MOG2':
    back_subtractor = cv2.createBackgroundSubtractorMOG2()
else:
    back_subtractor = cv2.createBackgroundSubtractorKNN()

# video_source = "/home/zpyang/Downloads/vtest1.mp4"
# video_source = "/home/zpyang/Downloads/vtest2.mp4"
# video_source = "/home/zpyang/Downloads/vtest-4hz.mp4"
# video_source = "/home/zpyang/Downloads/vtest-10hz.mp4"

video_source = 0

capture = cv2.VideoCapture(video_source)
if not capture.isOpened():
    print('Unable to open')
    exit(0)

while True:
    timer = cv2.getTickCount()
    ret, frame = capture.read()
    if frame is None:
        break        
    
    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    # kernal = np.ones((5, 5), "uint8") # squre kernal

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame = cv2.GaussianBlur(frame, (7, 7), 0)
    fg_mask = back_subtractor.apply(frame)

    th, fg_binary = cv2.threshold(fg_mask, 80, 255, cv2.THRESH_BINARY)
    fg_binary = cv2.dilate(fg_binary, kernal, iterations=1)

    blob_bboxes = [] # blob

    blob_thresh = 400

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)


    # cv2.putText(orig_frame, self.subtractor_name + " Subtractor",
    #             (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
    

    cv2.imshow('Motion Detection', frame)
    cv2.imshow("Motion Mask", fg_binary)
    cv2.imshow("FG Mask", fg_mask)
    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        exit(0)