import os
import numpy as np
import cv2

def find_if_close(cnt1,cnt2):
    row1,row2 = cnt1.shape[0],cnt2.shape[0]
    for i in xrange(row1):
        for j in xrange(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < 10 :
                return True
            elif i==row1-1 and j==row2-1:
                return False

cap = cv2.VideoCapture(os.getcwd() + "/their_football_videos/right_camera.mp4")
firstFrame = cv2.imread(os.getcwd()+ "/bg.jpg")

lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])
lower_red = np.array([0,140,110])
upper_red = np.array([20,200,255])
for i in range (0,50):
    _,frame = cap.read()
    frame = cv2.GaussianBlur(frame,(5,5),0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #find blue players
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(frame,frame, mask=mask)
    image = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #mask = cv2.inRange(hsv,lower_red, upper_red)
    #res = cv2.bitwise_and(frame,frame, mask=mask)
    _,thresh = cv2.threshold(image,25,255, cv2.THRESH_BINARY)
    (cnts, _) = cv2.findContours(thresh,cv2.cv.CV_RETR_TREE,cv2.cv.CV_CHAIN_APPROX_SIMPLE)
    for c in cnts:
        M = cv2.moments(c)
        if (M['m00'] != 0):
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            radius = cv2.arcLength(c,True)
            cv2.circle(frame, (cx,cy), 10,(255,0, 0), cv2.cv.CV_FILLED)
##    LENGTH = len(cnts)
##    status = []
##
##    for i in range(0,len(cnts)-1):
##        if i!= len(cnts)-1:
##            dist = find_if_close(cnts[i],cnts[i+1])
##            if dist == True:
##                val = np.minimum(cnts[i],cnts[i+1])
##                M = cv2.moments(val)
##                if (M['m00'] != 0):
##                    cx = int(M['m10']/M['m00'])
##                    cy = int(M['m01']/M['m00'])
##                    cv2.circle(frame, (cx,cy), 10,(255,0, 0), cv2.cv.CV_FILLED)
    #cv2.circle(frame, (cx,cy), 10,(255,0, 0), cv2.cv.CV_FILLED)
    #cv2.drawContours(thresh,unified,-1,255,-1)
    cv2.imshow("Blue players", frame)
    cv2.waitKey(30)
cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()
##    unified = []
##    maximum = int(status.max())+1
##    for i in xrange(maximum):
##        pos = np.where(status==i)[0]
##        if pos.size != 0:
##            cont = np.vstack(cnts[i] for i in pos)
##            hull = cv2.convexHull(cont)
##            unified.append(hull)

##    for i,cnt1 in enumerate(cnts):
##        x = i    
##        if i != LENGTH-1:
##            for j,cnt2 in enumerate(cnts[i+1:]):
##                x = x+1
##                dist = find_if_close(cnt1,cnt2)
##                if dist == True:
##                    val = min(status[i],status[x])
##                    status[x] = status[i] = val
##                else:
##                    if status[x]==status[i]:
##                        status[x] = i+1

##_,frame = cap.read()
##frameDelta = cv2.absdiff(firstFrame, frame)
##cv2.imshow("Difference", frameDelta)
##cv2.waitKey(0)
  
#thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

#thresh = cv2.dilate(thresh, None, iterations=2)
#(cnts, _) = cv2.findContours(thresh.copy(),cv2.cv.CV_RETR_TREE , cv2.cv.CV_CHAIN_APPROX_SIMPLE)
 
# loop over the contours
##for c in cnts:
##    ret,frame = cap.read()
##    # compute the bounding box for the contour, draw it on the frame,
##    # and update the text
##    (x, y, w, h) = cv2.boundingRect(c)
##    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
##
##    # show the frame and record if the user presses a key
##    cv2.imshow("Security Feed", frame)
##    #cv2.imshow("Thresh", thresh)
##    cv2.imshow("Frame Delta", frameDelta)
##    key = cv2.waitKey(1) & 0xFF
## 
##    # if the `q` key is pressed, break from the lop
##    if key == ord("q"):
##        break		     
##cv2.destroyAllWindows()
##cap.release()
