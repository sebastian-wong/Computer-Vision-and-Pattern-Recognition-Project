import os
import cv2
import cv2.cv as cv
import numpy as np

def convert2Integers(frame_width,frame_height,fps,frame_count):
    frame_width = int(frame_width)
    frame_height = int(frame_height)
    fps = int(fps)
    frame_count = int(frame_count)
    return frame_width, frame_height, fps, frame_count

def getBackgroundObject(cap, frame_count):
    _,img = cap.read()
    avgImg = np.float32(img)
    for fr_no in range(1,4140):
        _,img = cap.read()
        avgImg = (fr_no/(fr_no+1.0))*avgImg + (1.0/(fr_no+1.0))*img
        # convert into uint8 image
        normImg = cv2.convertScaleAbs(avgImg)
        cv2.imshow('img',img)
        cv2.imshow('normImg', normImg)
        cv2.waitKey(30)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()
    return normImg
    
cap = cv2.VideoCapture(os.getcwd() + "/Videos/stitchedVideo.avi")
frame_width = cap.get(cv.CV_CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv.CV_CAP_PROP_FPS)
frame_count = cap.get(cv.CV_CAP_PROP_FRAME_COUNT)
print frame_width, frame_height, fps, frame_count
#convert to integer form
frame_width, frame_height, fps, frame_count = convert2Integers(frame_width,frame_height,fps,frame_count)
#get background object
snapshot = getBackgroundObject(cap,frame_count)
cv2.imwrite(os.getcwd() + "/Results/background.jpg", snapshot)

