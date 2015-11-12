import numpy as np
import cv2
import os
     
cap = cv2.VideoCapture(os.getcwd() + "/stitchedVideo.mov")
firstFrame = cv2.imread(os.getcwd()+ "/background.jpg")

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 30,
                       qualityLevel = 0.2,
                       minDistance = 5,
                       blockSize = 5 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (5,5),
                  maxLevel = 30,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.03))

# Create some random colors
color = np.random.randint(0,255,(3000,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
#lower_reso = cv2.pyrDown(old_gray)
#lower_reso = cv2.pyrDown(lower_reso)
#pyold = np.zeros_like(old_frame)
#p0 = cv2.goodFeaturesToTrack(lower_reso, mask = None, **feature_params)
p0 = cv2.goodFeaturesToTrack(old_gray, **feature_params)

# Create a mask image for drawing purposes
#mask = np.zeros_like(lower_reso)
mask = np.zeros_like(old_frame)

cv2.namedWindow("Final", 0)
cv2.resizeWindow("Final", 100,100)

while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #lower_reso1 = cv2.pyrDown(lower_reso1)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray,frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        cv2.line(mask, (a,b),(c,d),color[i].tolist(), 2)
        cv2.circle(frame,(a,b),10,color[i].tolist(),-1)
    img = cv2.add(frame,mask)

    cv2.imshow("Final",frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
       break

    # Now update the previous frame and previous points
    #diff = diff1.copy()
    old_gray = frame_gray.copy()
    #lower_reso = lower_reso1.copy()
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()
