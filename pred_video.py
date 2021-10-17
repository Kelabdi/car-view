import cv2
import time


cap = cv2.VideoCapture("videos/ferrari.mp4")

while True:
    success, img = cap.read()
    cv2.imshow("Image", img)
    cv2.waitKey(1)
