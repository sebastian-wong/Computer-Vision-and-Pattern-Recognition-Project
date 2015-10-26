import os
import sys
import math
import numpy as np
import cv2
from matplotlib import pyplot as plt

from numpy import linalg

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

        # draw a rectangle around the region of interest
        #cv2.circle(img, refPt,2 , (0, 255, 0), 2)
        #cv2.imshow("image", img)

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
        if key == ord("u"):
            img=cv2.pyrUp(img)
        if key == ord("d"):
            img=cv2.pyrDown(img)    
            # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break
    cv2.destroyAllWindows()
            
# Storing good matches according to Lowe's ratio test
def filterMatches(matches, ratio = 0.75):
    filteredMatches = []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            filteredMatches.append(m[0])
    return filteredMatches

def imageDistance(matches):
    sumDistance = 0.0
    for match in matches:
        sumDistance += match.distance
    return sumDistance
    
    
def findDimensions(image, homography):
    base_p1 = np.ones(3, np.float32)
    base_p2 = np.ones(3, np.float32)
    base_p3 = np.ones(3, np.float32)
    base_p4 = np.ones(3, np.float32)

    (y, x) = image.shape[:2]

    base_p1[:2] = [0,0]
    base_p2[:2] = [x,0]
    base_p3[:2] = [0,y]
    base_p4[:2] = [x,y]

    max_x = None
    max_y = None
    min_x = None
    min_y = None

    for pt in [base_p1, base_p2, base_p3, base_p4]:

        hp = np.matrix(homography, np.float32) * np.matrix(pt, np.float32).T

        hp_arr = np.array(hp, np.float32)

        normal_pt = np.array([hp_arr[0]/hp_arr[2], hp_arr[1]/hp_arr[2]], np.float32)

        if ( max_x == None or normal_pt[0,0] > max_x ):
            max_x = normal_pt[0,0]

        if ( max_y == None or normal_pt[1,0] > max_y ):
            max_y = normal_pt[1,0]

        if ( min_x == None or normal_pt[0,0] < min_x ):
            min_x = normal_pt[0,0]

        if ( min_y == None or normal_pt[1,0] < min_y ):
            min_y = normal_pt[1,0]

    min_x = min(0, min_x)
    min_y = min(0, min_y)

    return (min_x, min_y, max_x, max_y)    


# stitch image1 and image2
def imageStitching(img1,img2):
    
    # Convert to grayscale for processing
    image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Use a SURF detector
    # Hessian Threshold at 5000
    detector = cv2.SURF()
    # Finding key points in first image
    image1Features, image1Descs = detector.detectAndCompute(image1,None)
    print image1Features
    # Parameters for nearest neighbour matching
    FLANN_INDEX_KDTREE = 1
    # Using Fast Approximate Nearest Neighbour Search Library
    flann_params = dict(algorithm = FLANN_INDEX_KDTREE,trees = 5)
    # Specifies number of times the trees in the index should be recursively
    # traversed.
    # Higher values has greater precision
    search_params = dict(checks=50)    
    matcher = cv2.FlannBasedMatcher(flann_params,search_params)
    
    # reduce image noise and detail
    image2GB = cv2.GaussianBlur(image2,(5,5),0)

    # Finding key points in second image
    image2Features, image2Descs = detector.detectAndCompute(image2GB,None)
    
    # Matching keypoints descriptors
    # Finding k best matches for each descriptor from a query set
    matches = matcher.knnMatch(image2Descs, trainDescriptors = image1Descs, k=2)
    print "Number of matches is ", len(matches)
    
    # Filtering close matches
    # indistinguisable
    filteredMatches = filterMatches(matches)
    print "Number of good matches is ", len(filteredMatches)
    distance = imageDistance(filteredMatches)
    print "Distance from image1 is " , distance
    
    averageDistance = distance/float(len(filteredMatches))
    print "Average Distance is ", averageDistance
    
    keyPoints1 = []
    keyPoints2 = []
    
    for match in filteredMatches:
        keyPoints1.append(image1Features[match.trainIdx])
        keyPoints2.append(image2Features[match.queryIdx])

    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = filteredMatches,
                       flags = 0)

    img3 = cv2.drawKeypoints(img1,keyPoints1,None,(255,0,0),4)

    plt.imshow(img3),plt.show()
    img4 = cv2.drawKeypoints(img2,keyPoints2,None,(255,0,0),4)

    plt.imshow(img4),plt.show()

    
    points1 = np.array([k.pt for k in keyPoints1])
    points2 = np.array([k.pt for k in keyPoints2])
    
    # finds perspective transformation between two planes
    # use Random sample consensus (RANSAC) based method
    # Maximum allowed reprojection error to treat a point pair as an inlier - 5
    # Levenberg-Marquardt method is also applied to reduce reprojection error
    homography, status = cv2.findHomography(points1,points2, cv2.RANSAC,5.0)
    print '%d / %d  inliers/matched' % (np.sum(status), len(status))    
    
    #inlierRatio = float(np.sum(status)) / float(len(status))
    
    homography = homography/homography[2,2]
    homographyInverse = linalg.inv(homography)
    
    
    (minimumX, minimumY, maximumX, maximumY) = findDimensions(image2GB, homographyInverse)
    
    # Adjust maximum x and y by image1 size
    maximumX = max(maximumX,image1.shape[1])
    maximumY = max(maximumY,image1.shape[0])
    
    moveH = np.matrix(np.identity(3),np.float32)
    
    if (minimumX < 0):
        moveH[0,2] += -minimumX
        maximumX += -minimumX
    if (minimumY < 0):
        moveH[1,2] += -minimumY
        maximumY += -minimumY
    
    print "minimum points is ", (minimumX, minimumY)
    print "Homography matrix is \n", homography
    print "Inverse Homography matrix is \n", homographyInverse
            
    # updating parameters        
    newHomographyInverse = moveH*homographyInverse
    newWidth = int(math.ceil(maximumX))
    newHeight = int(math.ceil(maximumY))
    
    print "new Inverse Homography matrix is \n", newHomographyInverse
    print "new width and height is ", (newWidth,newHeight)
    
    # Warping image1
    warpedImage1 = cv2.warpPerspective(img1,moveH,(newWidth,newHeight))
    # Warping image2
    warpedImage2 = cv2.warpPerspective(img2,newHomographyInverse,(newWidth,newHeight))
    
    # Creating new palette for image1 with enlarged height and width
    enlargedImage1 = np.zeros((newHeight,newWidth,3), np.uint8)
    
    # Create a mask from the warped image for constructing masked composite
    # Use simple binary thresholding
    # greater than 0 -> set to 255
    (ret,dataMap) = cv2.threshold(cv2.cvtColor(warpedImage2, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)
    
    # adding first image
    # Mask so that points in the second image will not be overwritten
    # 8 bit unsigned, 1 channel
    enlargedImage1 = cv2.add(enlargedImage1,warpedImage1,mask=np.bitwise_not(dataMap),dtype = cv2.CV_8U)
    
    # adding second image
    finalImage = cv2.add(enlargedImage1,warpedImage2,dtype = cv2.CV_8U)
    
    # Cropping
    finalImageGray = cv2.cvtColor(finalImage,cv2.COLOR_BGR2GRAY)
    # Need to find continuous points along the boundary that
    # has same colour or intensity
    # Using binary image before finding contours
    # Can apply canny edge detection instead
    ret, threshold = cv2.threshold(finalImageGray,1,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(threshold,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Can consider using cv2.CHAIN_APPROX_SIMPLE to get 4 points
    
    maximumArea = 0
    rectangle = (0,0,0,0)
    
    for contour in contours:
        area = 0
        x,y,xPrime,yPrime = cv2.boundingRect(contour)
        
        height = yPrime - y
        width = xPrime - x
        area = height * width
        
        # accounting for cases whereby there are
        # negative height and width
        if (area > maximumArea and height > 0 and width > 0):
            maximumArea = area
            rectangle = (x,y,xPrime,yPrime)
            
    # slicing    
    croppedFinalImage = finalImage[rectangle[1]:rectangle[1]+rectangle[3], rectangle[0]:rectangle[0]+rectangle[2]]
        
    return croppedFinalImage        
        

capLeft = cv2.VideoCapture(os.getcwd() + "/their_football_videos/left_camera.mov")
capCentre = cv2.VideoCapture(os.getcwd() + "/their_football_videos/centre_camera.mov")
capRight = cv2.VideoCapture(os.getcwd() + "/their_football_videos/right_camera.mov")

#For windows version
#capLeft = cv2.VideoCapture(os.getcwd() + "/their_football_videos/left_camera.mp4")
#capCentre = cv2.VideoCapture(os.getcwd() + "/their_football_videos/centre_camera.mp4")
#capRight = cv2.VideoCapture(os.getcwd() + "/their_football_videos/right_camera.mp4")

# Get total number of frames
frameCounts = int(capLeft.get(7))
retLeft, left = capLeft.read()
retCentre, centre = capCentre.read()
retRight, right = capRight.read()
#Method for getting keypoints by clicking
#kpts=getKeypoints(left)
leftCentre= imageStitching(centre,left)
combined= imageStitching(leftCentre,right)
cv2.imwrite("stitched.jpg",combined)    

