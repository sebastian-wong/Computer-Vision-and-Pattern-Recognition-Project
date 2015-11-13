import os
import numpy as np
import cv2
import copy

LENGTH = 1200
HEIGHT = 800
INPUT_FILE_NAME = "beforeWarpImage.jpg"
OUTPUT_FILE_NAME = "bird_eye_view_video.jpg"
# def find_if_close(cnt1,cnt2):
#    row1,row2 = cnt1.shape[0],cnt2.shape[0]
#    for i in xrange(row1):
#        for j in xrange(row2):
#            dist = np.linalg.norm(cnt1[i]-cnt2[j])
#            if abs(dist) < 10 :
#                return True
#            elif i==row1-1 and j==row2-1:
#                return False

# def removeOverlaps(cnts):
#     for i,cnt1 in enumerate(cnts):
#         if i < len(cnts)-2:
#             for j,cnt2 in enumerate(cnts[i+1:]):
#                 (x, y, w, h) = cv2.boundingRect(cnt1)
#                 (x1, y1, w1, h1) = cv2.boundingRect(cnt2)
#                 if((x1<x<x1+w1 and (y<y1<y+h or y<y1+h1<y+h or (y1+h1>y+h and y1<y)) or  ))
#
#
#
#                 if ((x1 > x and x1 < x+w ) or (x1+w1 > x and x1+w1 < x+w) or (y1+h1 > y and y1+h1 < y+h) or (y1 > y and y1<y+h)):
#                     cnts[j] = 0
#     return cnts

refPt=[]

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    global refPt
    if event == cv2.EVENT_LBUTTONDOWN:
        print "left click recorded"
        Pt = (x, y)
        refPt.append(Pt)
        print Pt

def getKeypoints(img):
    clone = img.copy()
    cv2.namedWindow("image",cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image", click_and_crop)
    flag=True 
    # keep looping until the 'c' key is pressed
    while flag:
            # display the image and wait for a keypress
        cv2.imshow("image", img)
        key = cv2.waitKey(1) & 0xFF
            # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            img = clone.copy()
            # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break
    cv2.destroyAllWindows()


#values for cropping 3126,116 and 4887,482 
cap = cv2.VideoCapture(os.getcwd() + "/stitchedVideo.avi")
firstFrame = cv2.imread(os.getcwd()+ "/football field.png")
#getKeypoints(firstFrame)
lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])
lower_red = np.array([0,140,110])
upper_red = np.array([20,200,255])
lower_white = np.array([100,0,100])
upper_white = np.array([200,200,200])
lower_yellow = np.array([30,150,200])
upper_yellow = np.array([100,255,255])

# lower_green = np.array([40,200,200])
# upper_green = np.array([71,255,255])
lower_green = np.array([30,150,200])
upper_green = np.array([100,255,255])


# lower_yellow = np.array([0,155,215])
# # bright green
# upper_yellow = np.array([100,255,255])
#cv2.namedWindow("Final", 0)
frameCounts = int(cap.get(7))
#cv2.resizeWindow("Final", 1200,500)
#height,width,layers = firstFrame.shape
#maskImg = np.ones(firstFrame.shape[:2], dtype="uint8") * 255
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
gradient=(482-116)/(4887-3126)*1.0
intecept=116-(3126*gradient)

#create the field dimension with the height
##field = np.zeros([LENGTH,HEIGHT])
##TopLeftPointBefore = (float(0), float(0))
##TopRightPointBefore = (float(LENGTH), float(0))
##BottomRightPointBefore = (float(LENGTH), float(HEIGHT))
##BottomLeftPointBefore = (float(0), float(HEIGHT))

#create the field dimension with the height
field = np.zeros([LENGTH,HEIGHT])
TopLeftPointBefore = (float(25), float(10))
TopRightPointBefore = (float(574), float(10))
BottomRightPointBefore = (float(574), float(365))
BottomLeftPointBefore = (float(25), float(365))

#hand picked points for final.jpg
TopLeftPointAfter = (float(1784),float(117))
TopRightPointAfter  = (float(3117), float(111))
BottomRightPointAfter  = (float(4866),float(488))
BottomLeftPointAfter  = (float(89), float(535))

fieldCoordsB = [TopLeftPointBefore, TopRightPointBefore, BottomRightPointBefore, BottomLeftPointBefore]
fieldCoordsA = [TopLeftPointAfter , TopRightPointAfter, BottomRightPointAfter, BottomLeftPointAfter]

frameCounts = int(cap.get(7))
print frameCounts
BEHomographyMatrix, status = cv2.findHomography(np.array(fieldCoordsA), np.array(fieldCoordsB), 0)
BEHomographyMatrix=np.array(BEHomographyMatrix,dtype='float32')
fourcc = cv2.cv.CV_FOURCC('D', 'I', 'V','X')
video = cv2.VideoWriter("bird_eye_view_stitchedVideo.avi",fourcc,24,(1200,800),True)

for i in range (0,100):

    redpoints=[]
    bluepoints=[]
    yellowpoints=[]        
    greenpoints=[]
    mappedbluepoints=[]
    mappedredpoints=[]
    mappedyellowpoints=[]
    mappedgreenpoints=[]
    _,frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5,5),np.uint8)
    firstFramecopy=firstFrame.copy()
        
    # find blue players
    maskBlue = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(frame,frame, mask=maskBlue)
    image = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(image, (33,33),1)
    # cv2.imshow('blur',blurred)
    # cv2.waitKey(0)
    dilation = cv2.dilate(blurred,kernel,iterations = 6)
    # cv2.imshow('dilation',dilation)
    # cv2.waitKey(0)
    _,threshBlue = cv2.threshold(dilation,10,255, cv2.THRESH_BINARY)
    (cntsBlue, _) = cv2.findContours(threshBlue,cv2.cv.CV_RETR_TREE,cv2.cv.CV_CHAIN_APPROX_SIMPLE)

    # find red players
    maskRed = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame,frame, mask=maskRed)
    image = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(image, (33,33),1)
    # cv2.imshow('blur',blurred)
    # cv2.waitKey(0)
    dilation = cv2.dilate(blurred,kernel,iterations = 6)
    # cv2.imshow('dilation',dilation)
    # cv2.waitKey(0)
    _,threshRed = cv2.threshold(dilation,10,255, cv2.THRESH_BINARY)
    (cntsRed, _) = cv2.findContours(threshRed,cv2.cv.CV_RETR_TREE,cv2.cv.CV_CHAIN_APPROX_SIMPLE)

    # find yellow players
    maskYellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    res = cv2.bitwise_and(frame,frame, mask=maskYellow)
    image = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(image, (33,33),1)
    dilation = cv2.dilate(blurred,kernel,iterations = 6)
    _,threshYellow = cv2.threshold(dilation,10,255, cv2.THRESH_BINARY)
    (cntsYellow, _) = cv2.findContours(threshYellow,cv2.cv.CV_RETR_TREE,cv2.cv.CV_CHAIN_APPROX_SIMPLE)
    
    # find green players
    maskGreen = cv2.inRange(hsv, lower_green, upper_green)
    res = cv2.bitwise_and(frame,frame, mask=maskGreen)
    image = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(image, (33,33),1)
    dilation = cv2.dilate(blurred,kernel,iterations = 6)
    _,threshGreen = cv2.threshold(dilation,10,255, cv2.THRESH_BINARY)
    (cntsGreen, _) = cv2.findContours(threshGreen,cv2.cv.CV_RETR_TREE,cv2.cv.CV_CHAIN_APPROX_SIMPLE)
    
                     
    for c in cntsBlue:
        # if cv2.contourArea(c) < 100:
        #     # move to next contour
        #     continue
        M = cv2.moments(c)
        if (M['m00'] != 0):
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            if cy >100 :
                #cv2.circle(frame, (cx,cy), 15,(255,0,0),2)
                bluepoints.append([cx,cy])
            
    for c in cntsRed:
        # if cv2.contourArea(c) < 5:
        #     # move to next contour
        #     continue
        M = cv2.moments(c)
        if (M['m00'] != 0):
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            if cy >100:
                #cv2.circle(frame, (cx,cy), 15,(0,0,255),2)
                redpoints.append([cx,cy])

    for c in cntsYellow:
        # if cv2.contourArea(c) < 200:
        #     # move to next contour
        #     continue
        M = cv2.moments(c)
        if (M['m00'] != 0):
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            if cy >100:
                #cv2.circle(frame, (cx,cy), 15,(0,255,255),2)
                yellowpoints.append([cx,cy])
            
    for c in cntsYellow:
        # if cv2.contourArea(c) < 200:
        #     # move to next contour
        #     continue
        M = cv2.moments(c)
        if (M['m00'] != 0):
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            if cy >100:
                #cv2.circle(frame, (cx,cy), 15,(0,255,0),2)
                greenpoints.append([cx,cy])

##    bluepoints=np.array(bluepoints,dtype='float32')
##    redpoints=np.array(redpoints,dtype='float32')
##    yellowpoints=np.array(yellowpoints,dtype='float32')
##    greenpoints=np.array(greenpoints,dtype='float32')

    for pts in bluepoints:
        pt=np.array([[[pts[0],pts[1]]]],dtype='float32')
        pt=cv2.perspectiveTransform(pt,BEHomographyMatrix)
        #mappedbluepoints.append(pt)
        cv2.circle(firstFramecopy, (pt[0][0][0],pt[0][0][1]), 15,(255,0,0),-1)

    for pts in redpoints:
        pt=np.array([[[pts[0],pts[1]]]],dtype='float32')
        pt=cv2.perspectiveTransform(pt,BEHomographyMatrix)
        #mappedredpoints.append(pt)
        cv2.circle(firstFramecopy, (pt[0][0][0],pt[0][0][1]), 15,(0,0,255),-1)

    for pts in yellowpoints:
        pt=np.array([[[pts[0],pts[1]]]],dtype='float32')
        pt=cv2.perspectiveTransform(pt,BEHomographyMatrix)
        #mappedyellowpoints.append(pt)
        cv2.circle(firstFramecopy, (pt[0][0][0],pt[0][0][1]), 15,(0,255,255),-1)

    for pts in greenpoints:
        pt=np.array([[[pts[0],pts[1]]]],dtype='float32')
        pt=cv2.perspectiveTransform(pt,BEHomographyMatrix)
        #mappedgreenpoints.append(pt)
        cv2.circle(firstFramecopy, (pt[0][0][0],pt[0][0][1]), 15,(0,255,0),-1)
            
    #afterWarpImage = cv2.warpPerspective(frame, BEHomographyMatrix , field.shape)
##    for pts in bluepoints:
##        cv2.circle(firstFrame, (pts[0],pts[1]), 15,(255,0,0),2)
##        
##    for pts in redpoints:
##        cv2.circle(firstFrame, (pts[0],pts[1]), 15,(0,0,255),2)
##
##    for pts in yellowpoints:
##        cv2.circle(firstFrame, (pts[0],pts[1]), 15,(0,255,255),2)
##        
##    for pts in greenpoints:
##        cv2.circle(firstFrame, (pts[0],pts[1]), 15,(0,255,0),2)
    
    resize = cv2.resize(firstFramecopy,(1200,800))
    if i%10 == 0:
        print i
        cv2.imwrite("AfterWarpImage"+str(i)+".jpg", firstFramecopy)
    video.write(firstFramecopy)
    
    #cv2.imshow("Final", resizedImage)
    cv2.waitKey(30)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("c"):
        break
cv2.imwrite("AfterWarpImage.jpg", firstFrame)

cv2.waitKey(0)
cv2.destroyAllWindows()
video.release()

cap.release()

