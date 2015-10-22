import os
import numpy as np
import cv2
from matplotlib import pyplot as plt


capLeft = cv2.VideoCapture(os.getcwd() + "/their_football_videos/left_camera.mp4")
capCentre = cv2.VideoCapture(os.getcwd() + "/their_football_videos/centre_camera.mp4")
capRight = cv2.VideoCapture(os.getcwd() + "/their_football_videos/right_camera.mp4")
ret, imageLeft = capLeft.read()
ret, imageCentre = capCentre.read()
#grayLeft = cv2.cvtColor(imageLeft, cv2.COLOR_BGR2GRAY)
#grayCentre = cv2.cvtColor(imageCentre, cv2.COLOR_BGR2GRAY)
r,c,ch = imageCentre.shape
imgarray = np.array([imageLeft,imageCentre])
outputarray = np.zeros([r,c,ch])
outputarray2 = np.array([imageLeft,imageCentre])
stitch = cv2.createStitcher()
stitch.estimateTransform(imgarray)
stitch.composePanorama(outputarray)
cv2.imwrite(os.getcwd() + "/trial.jpg" ,outputarray)
_ = stitch.stitch(imgarray,outputarray)
cv2.imwrite(os.getcwd() + "/trial2.jpg" ,outputarray)
