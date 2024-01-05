import cvzone
import cv2
import numpy as np

import math
from random import randint
# from cvzone.FaceDetectionModule import FaceDetector
from cvzone.HandTrackingModule import HandDetector

class SnakeGameClass:
    def __init__(self, food_path):
        self.points = [] # list of points on snake
        self.distances = [] # list of distances between each point
        self.curr_len = 0 # total length of snake rn
        self.allowed_len = 100 # total allowed len before eating food
        self.prev_head = 0, 0 # previous head point
        self.score = 0
        
        self.food_image = cv2.imread(food_path, cv2.IMREAD_UNCHANGED)
        self.food_height, self.food_width, _ = self.food_image.shape
        self.food_pos = 0, 0
        self.random_food_positions()
        
    def random_food_positions(self):
        pos = randint(100, 1000), randint(100, 600)
        while pos in self.points:
            pos = randint(100, 1000), randint(100, 600)
        self.food_pos = pos
        #print(pos)
    
    def update(self, img_main, curr_head):
        x_prev, y_prev = self.prev_head
        x_curr, y_curr = curr_head
        self.points.append([x_curr, y_curr])
        dist = math.hypot(x_prev - x_curr, y_prev - y_curr)
        self.distances.append(dist)
        self.curr_len += dist
        self.prev_head = curr_head
        
        if self.curr_len > self.allowed_len:
            for i, length in enumerate(self.distances):
                self.curr_len -= length
                self.distances.pop(i)
                self.points.pop(i)
                if self.curr_len < self.allowed_len: 
                    break
        rx, ry = self.food_pos
        rx -= self.food_width // 2
        ry -= self.food_height // 2
        img_main = cvzone.overlayPNG(img_main, self.food_image, (rx, ry))
                
        if abs(x_curr - rx) <= self.food_width and abs(y_curr - ry) <= self.food_height:
            print("apple has been eaten by the snake")
            self.random_food_positions()
            self.allowed_len += 30
            self.score += 1
        
        if curr_head in self.points[0:-1]:
            print("Collision")
        
        if self.points:
            for i, point in enumerate(self.points): # draw snake
                if i > 0:
                    cv2.line(img_main, self.points[i-1],self.points[i], color=(0, 0, 255), thickness=20)
            cv2.circle(img_main, self.points[-1], radius=20, color=(0,200,200),thickness=cv2.FILLED)
        
        
        
        '''hf, wf, cf = img_main.shape
        hb, wb, cb = self.food_image.shape

        x1, y1 = max(rx, 0), max(ry, 0)
        x2, y2 = min(rx + wf, wb), min(ry + hf, hb)

        # For -ve pos, change the starting pos in the overlay image
        x1_overlay = 0 if rx >= 0 else -rx
        y1_overlay = 0 if ry >= 0 else -ry

        # Calculate dim of slice to overlay
        wf, hf = x2 - x1, y2 - y1

        # If overlay is completely outside background, return original background
        if wf <= 0 or hf <= 0:
            return img_main

        # Extract the alpha channel from the foreground and create the inverse mask
        alpha = self.food_image[y1_overlay:y1_overlay + hf, x1_overlay:x1_overlay + wf, 3] / 255.0
        inv_alpha = 1.0 - alpha

        # Extract the RGB channels from the foreground
        imgRGB = self.food_image[y1_overlay:y1_overlay + hf, x1_overlay:x1_overlay + wf, 0:3]

        # Alpha blend the foreground and background
        for c in range(0, 3):
            img_main[y1:y2, x1:x2, c] = img_main[y1:y2, x1:x2, c] * inv_alpha + imgRGB[:, :, c] * alpha'''

        return img_main
    
    
cap = cv2.VideoCapture(0)
cap.set(3, 1280) # Increasing screen size from 640 x 480
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8, maxHands=1)
game = SnakeGameClass('food.png')

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
    if (cv2.waitKey(1) & 0xFF == ord('q')) or cv2.waitKey(1) == 27:
        break
