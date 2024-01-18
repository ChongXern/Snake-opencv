import cvzone
import cv2
import numpy as np

import time
import math
from random import randint
from mediapipe.tasks.python import vision
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
        self.gameOver = False
        
        
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
        if self.gameOver:
            cv2.putText(img_main, "GAME OVER", (640,360), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0,0,255),10,cv2.LINE_AA)
            time.sleep(3)
            self.gameOver = False
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
            print("SCORE:", self.score)
        
        if self.points:
            for i, point in enumerate(self.points): # draw snake
                if i > 0:
                    cv2.line(img_main, self.points[i-1],self.points[i], color=(0, 0, 255), thickness=20)
            cv2.circle(img_main, self.points[-1], radius=20, color=(0,200,200),thickness=cv2.FILLED)
        
        pts = np.array(self.points[:-2], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img_main, [pts], True, (0,200,0), 3)
        min_dist = cv2.pointPolygonTest(pts, curr_head,True)
        #print(abs(min_dist))
        if (abs(min_dist) <= 1): 
            self.points = [] # list of points on snake
            self.distances = [] # list of distances between each point
            self.curr_len = 0 # total length of snake rn
            self.allowed_len = 100 # total allowed len before eating food
            self.prev_head = 0, 0 # previous head point
            self.score = 0
            self.random_food_positions()
            self.gameOver = True
        
        '''if curr_head in self.points[0:-1]:
            print("Collision")
            self.gameOver = True'''
        
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
        cv2.putText(img, str(game.score), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255),3,cv2.LINE_AA)
        landmark_list = hands[0]['lmList']
        index_finger = landmark_list[8][0:2]
        img = game.update(img, index_finger)
    
    cv2.imshow("camera", img)
    # cv2.waitKey(1) # wait one millisecond
    if (cv2.waitKey(1) & 0xFF == ord('q')) or cv2.waitKey(1) == 27:
        break
