import os 
import cv2
import cv2.cv as cv
import numpy as np


cap = cv2.VideoCapture('test.mp4')

frame_width = cap.get(cv.CV_CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv.CV_CAP_PROP_FPS)
frame_count = cap.get(cv.CV_CAP_PROP_FRAME_COUNT)
print 'width:', frame_width
print 'height:', frame_height
print 'frames per second:', fps
print 'frame count:', frame_count

frame_width = int(frame_width)
frame_height = int(frame_height)
fps = int(fps)
frame_count = int(frame_count)

_,img = cap.read()
avgImg = np.float32(img)
for fr in range(1,frame_count):
    _,img = cap.read()
    alpha = 1 / float(fr + 1)
    cv2.accumulateWeighted(img, avgImg, alpha)
    normImg = cv2.convertScaleAbs(avgImg) # convert into uint8 image cv2.imshow('img',img)
    cv2.imshow('normImg', normImg)
    #print "fr = ", fr, " alpha = ", alpha
cv2.waitKey(0) 
cv2.destroyAllWindows()
cv2.imwrite('test_s.jpg', normImg)


cap.release()