import cvzone
import cv2
import numpy as np
import math
#from cvzone.FaceDetectionModule import FaceDetector
from cvzone.HandTrackingModule import HandDetector

class SnakeGameClass:
    def __init__(self):
        self.points = [] # list of points on snake
        self.distances = [] # list of distances between each point
        self.curr_len = 0 # total length of snake rn
        self.allowed_len = 150 # total allowed len before eating food
        self.prev_head = 0, 0 # previous head point
    
    def update(self, img_main, curr_head):
        x_prev, y_prev = self.prev_head
        x_curr, y_curr = curr_head
        self.points.append([x_curr, y_curr])
        dist = math.hypot(x_prev - x_curr, y_prev - y_curr)
        self.distances.append(dist)
        self.curr_len += dist
        self.prev_head = curr_head
    
        for i, point in enumerate(self.points): # draw snake
            if i > 0:
                cv2.line(img_main, self.points[i-1],self.points[i], color=(0, 0, 255), thickness=20)
        cv2.circle(img_main, self.points[-1], radius=20, color=(0,200,200),thickness=cv2.FILLED)
        
        return img_main
    
cap = cv2.VideoCapture(0)
cap.set(3, 1280) # Increasing screen size from 640 x 480
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8, maxHands=1)
game = SnakeGameClass()

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img,flipType=False)
    
    if hands:
        landmark_list = hands[0]['lmList']
        index_finger = landmark_list[8][0:2]
        img = game.update(img, index_finger)
    
    cv2.imshow("camera", img)
    # cv2.waitKey(1) # wait one millisecond
    # if cv2.waitKey(1) == 27:
    if (cv2.waitKey(1) & 0xFF == ord('q')) or cv2.waitKey(1) == 27:
        break
