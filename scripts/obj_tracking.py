#!/usr/bin/env python3
import numpy as np
import cv2
import util
import rospy
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
from visnet.msg import CamMsmt

# Use calibrated data to undistort bounding boxes
import os
data_dir = os.path.abspath( os.path.join(os.path.dirname(__file__), os.pardir)) + "/data/" 
info_dir = os.path.abspath( os.path.join(os.path.dirname(__file__), os.pardir)) + "/camera_info/"
import yaml

cam_names = [
    "camera_0",
    "camera_1",
    # "camera_2",
    "camera_3",
    ]

# load calibration and camera poses
cam_pinhole_Ks = {}
cam_Ps = {}
cam_dists = {}
for i, cam_name in enumerate(cam_names):
    cam_info = {}
    with open(f"{info_dir}/{cam_name}.yaml", 'r') as fs: 
        cam_info = yaml.safe_load(fs)
    cam_pinhole_K = np.array(cam_info["camera_matrix"]["data"])
    cam_dist = np.array(cam_info["distortion_coefficients"]["data"])
    cam_P, dist_valid_roi = cv2.getOptimalNewCameraMatrix(np.array(cam_pinhole_K).reshape(3,3), np.array(cam_dist), (1600,1200), 1, (1600, 1200))
    cam_Ps[cam_name] = cam_P
    cam_pinhole_Ks[cam_name] = cam_pinhole_K
    cam_dists[cam_name] = cam_dist

# For nanodet
import os
import torch
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight
from nanodet.data.transform import Pipeline
from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.util import overlay_bbox_cv

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device('cuda')
torch.backends.cudnn.enabled = True
nanodet_dir = os.path.abspath( os.path.join(os.path.dirname(__file__), os.pardir)) + "/nanodet/" 

config_path = nanodet_dir+'config/nanodet-plus-m-1.5x_416-cf.yml'
model_path = nanodet_dir+'trained/nanodet_model_best.pth'
load_config(cfg, config_path)

class MotionDetector:
    def __init__(self, subtractor_name=None, tracker_name=None, view=False):
        if subtractor_name is not None:
            if subtractor_name == 'MOG2':
                self.back_subtractor = cv2.createBackgroundSubtractorMOG2()
            else:
                self.back_subtractor = cv2.createBackgroundSubtractorKNN()
            self.subtractor_name = subtractor_name

        self.tracker_name = tracker_name
        self.timer = None
        self.bboxes = None
        self.view = view
        self.last_mask = None

        self.tracker_timeout = 0.5 # kill tracker after failing to track
        self.trackers = []
        self.tracker_timers = []
        self.not_ok_time_list = []

    def tracker_init(self, frame, pred_bbox):
        print("init bbox: ", pred_bbox)
        tracker_name = self.tracker_name
        if tracker_name is not None:
            # tracker_names = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
            tracker = None
            if tracker_name == 'BOOSTING': # old and bad
                tracker = cv2.TrackerBoosting_create()
            elif tracker_name == 'MIL':
                tracker = cv2.TrackerMIL_create()
            elif tracker_name == 'KCF': # recommend to use this for most cases
                tracker = cv2.TrackerKCF_create()
            elif tracker_name == 'TLD':
                tracker = cv2.TrackerTLD_create()
            elif tracker_name == 'MEDIANFLOW':
                tracker = cv2.TrackerMedianFlow_create()
            elif tracker_name == 'GOTURN': # needs extra files to config CNN
                tracker = cv2.TrackerGOTURN_create()
            elif tracker_name == 'MOSSE':
                tracker = cv2.legacy.TrackerMOSSE_create()
            elif tracker_name == "CSRT":
                tracker = cv2.TrackerCSRT_create()
            else:
                print("Please check tracker name!")
            
            if tracker is not None:
                tracker_started = tracker.init(frame, pred_bbox)
                self.tracker_timers.append(cv2.getTickCount())
                self.trackers.append(tracker)
                self.not_ok_time_list.append(0)

    def tracker_remove(self, i):
        self.trackers.remove(i)
        self.tracker_timers.remove(i)
        self.not_ok_time_list.remove(i)
    
    def track_2d(self, frame):
        bboxes = []

        for i, tracker in enumerate(self.trackers):
            self.tracker_timers[i] = cv2.getTickCount()
            OK, bbox = tracker.update(frame)

            fps = cv2.getTickFrequency() / (cv2.getTickCount() - self.tracker_timers[i])

            if OK:
                #tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
                cv2.putText(frame, self.tracker_name + " Tracker", (100, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
                bboxes.append(bbox)
            else:
                #tracking failed
                cv2.putText(frame, "Tracking failure detected", (100, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                self.not_ok_time_list[i] += 1/fps

        for i, dur in enumerate(self.not_ok_time_list):
            if dur > self.tracker_timeout:
                self.tracker_remove(i)
        
        cv2.imshow("tracker2D", frame)

        return bboxes

    def detect(self, frame):
        """
        Detects moving objects in the frame, and returns the bounding boxes
        """
        self.timer = cv2.getTickCount()

        orig_frame = frame.copy()

        kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
        # kernal = np.ones((5, 5), "uint8") # squre kernal
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        fg_mask = self.back_subtractor.apply(frame)

        th, fg_binary = cv2.threshold(fg_mask, 80, 255, cv2.THRESH_BINARY)
        fg_binary = cv2.dilate(fg_binary, kernal, iterations=2)

        if self.last_mask is not None:
            motion_mask = cv2.bitwise_and(fg_binary, self.last_mask)
        else:
            motion_mask = fg_binary
        
        self.last_mask = fg_binary
        blob_thresh = 400
        
        bboxes = []

        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if w * h > blob_thresh:
                bboxes.append([x,y,w,h])
                cv2.rectangle(orig_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - self.timer)

        if self.view:
            # cv2.putText(orig_frame, self.subtractor_name + " Subtractor",
            #             (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            # Display FPS on frame
            cv2.putText(orig_frame, "FPS : " + str(int(fps)), (100, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            # cv2.imshow("original", orig_frame)

            # cv2.imshow("Binary Mask", fg_binary)
            # cv2.imshow("Motion Mask", motion_mask)
            # cv2.imshow("FG Mask", fg_mask)
            # print(k)
            return bboxes, orig_frame      
        return bboxes, None

class Predictor(object):
    def __init__(self, cfg, model_path, logger, device="cuda:0"):
        self.cfg = cfg
        self.device = device
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)
        if cfg.model.arch.backbone.name == "RepVGG":
            deploy_config = cfg.model
            deploy_config.arch.backbone.update({"deploy": True})
            deploy_model = build_model(deploy_config)
            from nanodet.model.backbone.repvgg import repvgg_det_model_convert

            model = repvgg_det_model_convert(model, deploy_model)
        self.model = model.to(device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def inference(self, img):
        img_info = {"id": 0}
        img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)
        with torch.no_grad():
            results = self.model.inference(meta)
        return meta, results
    
    def extract_bbox(self, dets, class_names, score_thresh):
        # bounding box from inference (x0, y0, x1, y1, score)
        all_box = []
        for label in dets:
            for bbox in dets[label]:
                score = bbox[-1]
                if score > score_thresh:
                    x0, y0, x1, y1 = [int(i) for i in bbox[:4]]
                    all_box.append([x0, y0, x1-x0, y1-y0, score]) #(x, y, w, h, score)
        all_box.sort(key=lambda v: v[-1])
        return all_box

def img_cb(data, args):
    bridge: CvBridge = args[0]
    motion_det: MotionDetector = args[1]
    predictor: Predictor = args[2]
    msmt_pub: rospy.Publisher = args[3]
    img_pub: rospy.Publisher = args[4]
    cam_name = args[5]

    if data._type == "sensor_msgs/CompressedImage":
        cv_img = bridge.compressed_imgmsg_to_cv2(data, "bgr8")
    elif data._type == "sensor_msgs/Image":
        cv_img = bridge.imgmsg_to_cv2(data, "bgr8")
    
    motion_bbox, img = motion_det.detect(cv_img)
    meta, pred_res = predictor.inference(cv_img)
    pred_img = overlay_bbox_cv(cv_img.copy(), pred_res[0], cfg.class_names, score_thresh=0.35)
    # cv2.imshow("prediction result", pred_img)
    img_pub.publish(bridge.cv2_to_imgmsg(pred_img))
    
    pred_bbox = predictor.extract_bbox(pred_res[0], cfg.class_names, score_thresh=0.35)
    pred_bbox = np.array(pred_bbox, dtype=np.int16)
    motion_bbox = np.array(motion_bbox, dtype=np.int16)
    
    if pred_bbox.shape[0]:
        pred_bbox = pred_bbox[:, 0:-1] # don't send prediction score
        pred_bbox = undistort(pred_bbox, cam_name)
        pred_bbox = pred_bbox.ravel()
    
    if motion_bbox.shape[0]:
        motion_bbox = undistort(motion_bbox, cam_name)
        motion_bbox = motion_bbox.ravel()
 
    n_pred = int(pred_bbox.shape[0]/4)
    n_motion = int(motion_bbox.shape[0]/4)
    # print(f"num of inference [{n_pred}] | num of motion det [{n_motion}]")
    
    labels = np.concatenate([np.ones(n_pred), -np.ones(n_motion)])
    labels = labels.astype(np.int16)
    msmts = np.concatenate([pred_bbox, motion_bbox])

    pub_data = CamMsmt()
    pub_data.labels = labels.tolist()
    pub_data.msmts = msmts.tolist()
    msmt_pub.publish(pub_data)

    # print(pred_bbox)
    # track_bbox = motion_det.track_2d(cv_img)
    # print(motion_det.not_ok_time_list)

    # # logic to initialize 2d tracker from opencv
    # if pred_bbox: # classifier got hit(s)
    #     if not track_bbox: # no tracker is initialized or no trackin succeded
    #         for pbox in pred_bbox:
    #             x = pbox[0]
    #             y = pbox[1]
    #             w = pbox[2] - pbox[0]
    #             h = pbox[3] - pbox[1]
    #             motion_det.tracker_init(cv_img, (x, y, w, h))
    #     else:
    #         for pbox in pred_bbox:
    #             for i, tbox in enumerate(track_bbox):
    #                 iou = get_iou(pbox[0:-1], (tbox[0], tbox[1], tbox[0]+tbox[2], tbox[1]+tbox[3]))
    #                 print("iou: ", iou)
    #                 if iou < 0.0:
    #                     x = pbox[0]
    #                     y = pbox[1]
    #                     w = pbox[2] - pbox[0]
    #                     h = pbox[3] - pbox[1]
    #                     motion_det.tracker_init(cv_img, (x, y, w, h))
    #                     motion_det.tracker_remove(i)

    # keep the cv windows open
    # k = cv2.waitKey(3) & 0xff

def undistort(bboxes, cam_name):
    points = bboxes[:,0:2].astype(np.float32)
    undistort_points = cv2.undistortPoints(
        np.expand_dims(points, axis=1), 
        cam_pinhole_Ks[cam_name].reshape(3,3), 
        cam_dists[cam_name], 
        P=cam_Ps[cam_name])
    undistort_points = undistort_points.reshape(undistort_points.shape[0], undistort_points.shape[2])
    # print(f"{undistort_points.shape} | {bboxes[:,2:4].shape}")
    return np.hstack([undistort_points, bboxes[:,2:4]]).astype(np.int16)

def main(args):
    rospy.init_node("motion_detection")
    if args.mode == "single":
        cam_names = [args.cam]
    elif args.mode == "multi":
        cam_names = [
        "camera_0",
        "camera_1",
        # "camera_2",
        "camera_3",
        ]
    else:
        rospy.logerr("Invalid mode, must be (single) or (multi)!")
        return

    bridge = CvBridge()
    logger = Logger(-1, use_tensorboard=False)

    predictor = Predictor(cfg, model_path, logger, device=device)
    for cam_name in cam_names:
        print(f"setting up tracker for {cam_name}")

        motion_det = MotionDetector("MOG2", "KCF", view=True)

        msmt_pub_ = rospy.Publisher(f"{cam_name}/msmt", CamMsmt, queue_size=2)
        img_pub_ = rospy.Publisher(f"{cam_name}/tracked_view", Image, queue_size=1)

        img_cb_args = [bridge, motion_det, predictor, msmt_pub_, img_pub_, cam_name]
        if args.mode == "single":
            img_sub_ = rospy.Subscriber(f"{cam_name}/image_raw", Image, img_cb, img_cb_args)
        elif args.mode == "multi":
            img_sub_ = rospy.Subscriber(f"{cam_name}/image_raw/compressed/compressed", CompressedImage, img_cb, img_cb_args)

    rospy.spin()

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser("2D motion tracking/inference")
    parser.add_argument(
        "mode",
        default="single",
        help="subscribe to a single camera or multiple cameras"
    )
    
    parser.add_argument(
        "cam",
        default="camera",
        help="camera name (str)"
    )
    parser.add_argument("__name")
    parser.add_argument("__log")

    try:
        args = parser.parse_args()
        main(args)

    except Exception as e:
        print(e)