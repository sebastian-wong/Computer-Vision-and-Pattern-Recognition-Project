import os
import numpy as np
import cv2
import copy

kernel = np.ones((5,5),np.uint8)   
img = cv2.imread(os.getcwd() + "/goalkeeper.png")
blurred = cv2.GaussianBlur(img, (33,33),1)
cv2.imshow('blur',blurred)
cv2.waitKey(0)
dilation = cv2.dilate(blurred,kernel,iterations = 6)
cv2.imshow('dilation',dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("dilatedgoal.png",dilation)
cv2.waitKey(0)
