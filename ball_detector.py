from tracknet import BallTrackerNet
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial import distance
from tqdm import tqdm
from UNet import UNet

class BallDetector:
    def __init__(self, model_type='tracknet', device='cuda'):
        self.model_type = model_type
        self.device = device        
        
        if self.model_type == 'tracknet' or self.model_type == 'unet':
            if self.model_type == 'tracknet':
                self.model = BallTrackerNet(input_channels=9, out_channels=256)
                self.path_model = './models/model_ball_det_tracknet.pt'
                # print('using tracknet')
            else:
                self.model = UNet(input_channels=9, out_channels=256)
                self.path_model = './models/model_ball_det_unet.pt'
                # print('using unet')
            self.model.load_state_dict(torch.load(self.path_model, map_location=device))
            self.model = self.model.to(device)
            self.model.eval()
        elif self.model_type == 'yolo':
            self.detection_model = YOLO('models/yolo5_last.pt')
            self.detection_model = self.detection_model.to(device)
        else:
            raise ValueError("model_type should be 'yolo' or 'tracknet'")
        
        print(f"using {self.model_type}")
        self.width = 640
        self.height = 360

    def infer_model(self, frames):
        """ Run pretrained model on a consecutive list of frames
        :params
            frames: list of consecutive video frames type=np.ndarray shape=(720,  1280, 3) 
        :return
            ball_track: list of detected ball points (x, y)
        """
        ball_track = [(None, None)]*2
        prev_pred = [None, None]
        x_pred_last, y_pred_last = 0, 0
        for num in tqdm(range(2, len(frames))):
            if self.model_type == 'tracknet' or self.model_type == 'unet':
                img = cv2.resize(frames[num], (self.width, self.height)) # img.shape = (360, 640, 3)
                img_prev = cv2.resize(frames[num-1], (self.width, self.height))
                img_preprev = cv2.resize(frames[num-2], (self.width, self.height))
                imgs = np.concatenate((img, img_prev, img_preprev), axis=2)  # img.shape = (360, 640, 9)
                imgs = imgs.astype(np.float32)/255.0
                imgs = np.rollaxis(imgs, 2, 0)  # img.shape = (9, 360, 640)
                inp = np.expand_dims(imgs, axis=0) # inp.shape = (1, 9, 360, 640)

                out = self.model(torch.from_numpy(inp).float().to(self.device)) # out.shape=torch.Size([1, 256, 360, 640])
                output = out.argmax(dim=1).detach().cpu().numpy() # (1, 360, 640)
                x_pred, y_pred = self.postprocess(output, prev_pred)
                prev_pred = [x_pred, y_pred]
                ball_track.append((x_pred, y_pred))
            elif self.model_type == 'yolo':
                BALL_LABEL = 0 
                ball_min_score = 0.15
                results = self.detection_model(frames[num])
                preds = results[0]

                ball_boxes = []
                for box in preds.boxes:
                    if box.cls == BALL_LABEL and box.conf >= ball_min_score:
                        ball_boxes.append(box.xyxy.cpu().numpy()[0])
                        ball_min_score = box.conf
                
                if len(ball_boxes) > 0:
                    x_pred = int((ball_boxes[-1][2] + ball_boxes[-1][0]) / 2)
                    y_pred = int((ball_boxes[-1][3] + ball_boxes[-1][1]) / 2)
                    ball_track.append((x_pred, y_pred))
                    x_pred_last, y_pred_last = x_pred, y_pred
                else:
                    ball_track.append((x_pred_last, y_pred_last))

        return ball_track

    def postprocess(self, feature_map, prev_pred, scale=2, max_dist=80):
        """
        :params
            feature_map: feature map with shape (1,360,640)
            prev_pred: [x,y] coordinates of ball prediction from previous frame
            scale: scale for conversion to original shape (720,1280)
            max_dist: maximum distance from previous ball detection to remove outliers
        :return
            x,y ball coordinates
        """
        feature_map *= 255
        feature_map = feature_map.reshape((self.height, self.width))
        feature_map = feature_map.astype(np.uint8)
        ret, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY) # 将灰度图像进行二值化处理
        circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2,
                                   maxRadius=7) #检测图像中的圆圈 circles.shape=(1, N, 3)
        x, y = None, None
        if circles is not None:
            if prev_pred[0]:
                for i in range(len(circles[0])):
                    x_temp = circles[0][i][0]*scale
                    y_temp = circles[0][i][1]*scale
                    dist = distance.euclidean((x_temp, y_temp), prev_pred)
                    if dist < max_dist:
                        x, y = x_temp, y_temp
                        break                
            else:
                x = circles[0][0][0]*scale
                y = circles[0][0][1]*scale
        # else:
            # print("circles is None")
        return x, y
