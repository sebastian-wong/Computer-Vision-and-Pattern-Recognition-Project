import os
import numpy as np
import cv2
import copy

LENGTH = 1200
HEIGHT = 800
INPUT_FILE_NAME = "beforeWarpImage.jpg"
OUTPUT_FILE_NAME = "bird_eye_view_video.jpg"

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
        print "pixel value is"
        b=frame[y,x]
        print b

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
cap = cv2.VideoCapture(os.getcwd() + "/stitchedVideo.mov")
#cap = cv2.VideoCapture(os.getcwd() + "/their_football_videos/right_camera.mp4")
firstFrame = cv2.imread(os.getcwd()+ "/football_field.jpg")
background= cv2.imread(os.getcwd()+"/background.jpg")
# lower_blue = np.array([110,50,50])
# upper_blue = np.array([130,255,255])
lower_blue = np.array([100,50,50])
upper_blue = np.array([140,255,255])
lower_red = np.array([0,140,90])
upper_red = np.array([20,200,255])
# lower_red = np.array([0,140,110])
# upper_red = np.array([20,200,255])
lower_white = np.array([100,90,100])
upper_white = np.array([150,120,140])
lower_yellow = np.array([30,150,200])
upper_yellow = np.array([100,255,255])
lower_green = np.array([45, 100, 100])
upper_green = np.array([55, 120,160])
# lower_ball = np.array([30,56,175])
# upper_ball = np.array([50,71,183])
lower_ball = np.array([27,56,175])
upper_ball = np.array([50,71,183])
# upper_ball = np.array([31,94,102])

frameCounts = int(cap.get(7))
gradient=(482-116)/(4887-3126)*1.0
intecept=116-(3126*gradient)

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
video = cv2.VideoWriter("bird_eye_view_mappedVideo.avi",fourcc,24,(600,376),True)

for i in range (0,1000):

    redpoints=[]
    bluepoints=[]
    yellowpoints=[]        
    greenpoints=[]
    whitepoints=[]
    ballpoints = []
    
    mappedbluepoints=[]
    mappedredpoints=[]
    mappedyellowpoints=[]
    mappedgreenpoints=[]
    _,frame = cap.read()

    diff=cv2.absdiff(frame,background)
    diff_hsv=cv2.cvtColor(diff,cv2.COLOR_BGR2HSV)
    _,_,value=cv2.split(diff_hsv)
    _,thresAbsDiff=cv2.threshold(value,40,255, cv2.THRESH_BINARY)
    res=cv2.bitwise_and(frame,frame,mask=thresAbsDiff)
    hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5,5),np.uint8)
    firstFramecopy=firstFrame.copy()
    
    # find ball
    maskBall = cv2.inRange(hsv, lower_ball, upper_ball)
    res = cv2.bitwise_and(frame,frame, mask=maskBall)
    # cv2.imshow('res', res)
    # cv2.waitKey(0)
    h,s,v = cv2.split(res)
    # cv2.imshow('h',h)
    # cv2.waitKey(0)
    blurred = cv2.GaussianBlur(h, (33,33),1)
    dilation = cv2.dilate(blurred,kernel,iterations = 6)
    _,threshBall = cv2.threshold(dilation,25,255, cv2.THRESH_BINARY)
    (cntsBall, _) = cv2.findContours(threshBall,cv2.cv.CV_RETR_TREE,cv2.cv.CV_CHAIN_APPROX_SIMPLE)
    
    # find blue players
    maskBlue = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(hsv,hsv, mask=maskBlue)    
    image = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # res = cv2.bitwise_and(frame,frame, mask=maskBlue)
    # image = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(image, (33,33),1)
    dilation = cv2.dilate(image,kernel,iterations = 5)
    _,threshBlue = cv2.threshold(dilation,25,255, cv2.THRESH_BINARY)
    (cntsBlue, _) = cv2.findContours(threshBlue,cv2.cv.CV_RETR_TREE,cv2.cv.CV_CHAIN_APPROX_SIMPLE)

    # find red players
    maskRed = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(hsv,hsv, mask=maskRed)
    image = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # res = cv2.bitwise_and(frame,frame, mask=maskRed)
    # image = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(image, (33,33),1)
    dilation = cv2.dilate(blurred,kernel,iterations = 5)
    _,threshRed = cv2.threshold(dilation,25,255, cv2.THRESH_BINARY)
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
    
    # find white players
    diff = cv2.absdiff(frame,background)    
    maskWhite = cv2.inRange(diff, lower_white, upper_white)
    res = cv2.bitwise_and(diff,diff, mask=maskWhite)
    image = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(image, (33,33),3)
    dilation = cv2.dilate(blurred,kernel,iterations = 6)
    _,threshWhite = cv2.threshold(dilation,6,255, cv2.THRESH_BINARY)
    (cntsWhite, _) = cv2.findContours(threshWhite,cv2.cv.CV_RETR_TREE,cv2.cv.CV_CHAIN_APPROX_SIMPLE)
    
    
    # for c in cntsBall:
    #     M = cv2.moments(c)
    #     result = cv2.approxPolyDP(c,0.1,True)
    #     rows,columns,pointValues = result.shape
    #     # rows,columns = result.shape
    #     # if cv2.contourArea(c) > 675 and (rows<4 and rows>6):
    #     if cv2.contourArea(c) > 675 or (rows<4 and rows>6):
    #         # move to next contour
    #         continue
    #     if (M['m00'] != 0):
    #         cx = int(M['m10']/M['m00'])
    #         cy = int(M['m01']/M['m00'])
    #         if cy >100 :
    #             # cv2.circle(frame, (cx,cy), 15,(0,0,0),1)
    #             ballpoints.append([cx,cy])
    #             ballpt = sorted(ballpoints)[-1]
    # cv2.circle(frame, (ballpt[0],ballpt[1]), 15,(0,0,0),1)
    # print "ball " + str(i)
    # print cx,cy
    # print rows

                     
    for c in cntsBlue:
        M = cv2.moments(c)
        if (M['m00'] != 0):
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            if cy >100 :
                cv2.circle(frame, (cx,cy), 15,(255,0,0),2)
                bluepoints.append([cx,cy])
            
    for c in cntsRed:
        M = cv2.moments(c)
        if (M['m00'] != 0):
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            if cy >100:
                cv2.circle(frame, (cx,cy), 15,(0,0,255),2)
                redpoints.append([cx,cy])

    for c in cntsYellow:
        M = cv2.moments(c)
        if (M['m00'] != 0):
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            if cy >100:
                yellowpoints.append([cx,cy])
            
    for c in cntsGreen:
        # if cv2.contourArea(c) < 200:
        #     # move to next contour
        #     continue
        M = cv2.moments(c)
        if (M['m00'] != 0):
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            if cy >100:
                greenpoints.append([cx,cy])
                   
    for c in cntsWhite:
        M = cv2.moments(c)
        if cv2.contourArea(c) >1000:
            if (M['m00'] != 0):
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                if (175<cx<2542 and 116<cy<519):
                    whitepoints.append([cx,cy])
                                          
    redPlayers = []
    bluePlayers = []            

    for pts in bluepoints:
        pt=np.array([[[pts[0],pts[1]]]],dtype='float32')
        pt=cv2.perspectiveTransform(pt,BEHomographyMatrix)
        #mappedbluepoints.append(pt)
        if (pt[0][0][0]>20 and pt[0][0][0]<579 and 10<pt[0][0][1]<365):
            bluePlayers.append([pt[0][0][0],pt[0][0][1]])
            bluePlayers = sorted(bluePlayers)
            cv2.circle(firstFramecopy, (pt[0][0][0],pt[0][0][1]), 5,(255,0,0),-1)

    for pts in redpoints:
        pt=np.array([[[pts[0],pts[1]]]],dtype='float32')
        pt=cv2.perspectiveTransform(pt,BEHomographyMatrix)
        if (pt[0][0][0]>20 and pt[0][0][0]<579 and 10<pt[0][0][1]<365):
            redPlayers.append([pt[0][0][0],pt[0][0][1]])
            redPlayers = sorted(redPlayers)    
            cv2.circle(firstFramecopy, (pt[0][0][0],pt[0][0][1]), 5,(0,0,255),-1)

    for pts in yellowpoints:
        pt=np.array([[[pts[0],pts[1]]]],dtype='float32')
        pt=cv2.perspectiveTransform(pt,BEHomographyMatrix)
        if (pt[0][0][0]>20 and pt[0][0][0]<579):
            cv2.circle(firstFramecopy, (pt[0][0][0],pt[0][0][1]), 5,(0,255,255),-1)

    for pts in greenpoints:
        pt=np.array([[[pts[0],pts[1]]]],dtype='float32')
        pt=cv2.perspectiveTransform(pt,BEHomographyMatrix)
        if (pt[0][0][0]>475 and pt[0][0][0]<600 and 10<pt[0][0][1]<365):
            cv2.circle(firstFramecopy, (pt[0][0][0],pt[0][0][1]), 5,(0,255,0),-1)
            
    for pts in whitepoints:
        pt=np.array([[[pts[0],pts[1]]]],dtype='float32')
        pt=cv2.perspectiveTransform(pt,BEHomographyMatrix)
        if (pt[0][0][0]>20 and pt[0][0][0]<579 and 10<pt[0][0][1]<365):
            cv2.circle(firstFramecopy, (pt[0][0][0],pt[0][0][1]), 5,(255,255,255),-1)        


    # for offsight
    # left goalpost belongs to red players
    redOffSightX = redPlayers[0][0]
    redOffSight = ((redOffSightX,0),(redOffSightX,4902))
    cv2.line(firstFramecopy,redOffSight[0],redOffSight[1],(0,0,255),thickness = 1)
    
    # right goalpost belongs to blue players
    blueOffSightX = bluePlayers[-1][0]
    blueOffSight = ((blueOffSightX,0),(blueOffSightX,4902))
    cv2.line(firstFramecopy,blueOffSight[0],blueOffSight[1],(255,0,0),thickness = 1)

        
    resize = cv2.resize(firstFramecopy,(600,376))
    if i%10 == 0:
        print i

    cv2.imshow("Final", resize)
    # cv2.imshow("ballie",frame)
    
    cv2.waitKey(0)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("c"):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
video.release()
cap.release()

