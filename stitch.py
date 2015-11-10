import os
import sys
import math
import numpy as np
import cv2
import cv2.cv as cv
from matplotlib import pyplot as plt
from operator import itemgetter

from numpy import linalg


refPt=[]

# Storing good matches according to Lowe's ratio test
def filterMatches(matches, ratio = 0.75):
    filteredMatches = []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            filteredMatches.append(m[0])
    return filteredMatches

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

# given two images
# calculate their homography matrix:
def getHomography(img1,img2):
    # Convert to grayscale for processing
    image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Use a SURF detector
    # Hessian Threshold at 5000
    detector = cv2.SURF()
    detector.hessianThreshold = 2500
    # Finding key points in first image
    image1Features, image1Descs = detector.detectAndCompute(image1,None)
    
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
    print homography
    return homography
    
def stitching(image1,image2,homography):     
    homographyInverse = linalg.inv(homography)
    image2GB = cv2.GaussianBlur(image2,(5,5),0)

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
    
    #print "minimum points is ", (minimumX, minimumY)
    #print "Homography matrix is \n", homography
    #print "Inverse Homography matrix is \n", homographyInverse
            
    # updating parameters        
    newHomographyInverse = moveH*homographyInverse
    newWidth = int(math.ceil(maximumX))
    newHeight = int(math.ceil(maximumY))
    
    #print "new Inverse Homography matrix is \n", newHomographyInverse
    #print "new width and height is ", (newWidth,newHeight)
    
    # Warping image1
    warpedImage1 = cv2.warpPerspective(image1,moveH,(newWidth,newHeight))
    # Warping image2
    warpedImage2 = cv2.warpPerspective(image2,newHomographyInverse,(newWidth,newHeight))
    
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
    cv2.imwrite("cropped.jpg", croppedFinalImage)
    return croppedFinalImage,moveH,homographyInverse          

# given three videos and a time in seconds
# return the left, centre, and right frame
# of the video at the specified time
def getFramesAtSpecificTime(seconds):
    capLeft = cv2.VideoCapture(os.getcwd() + "/their_football_videos/left_camera.mov")
    capCentre = cv2.VideoCapture(os.getcwd() + "/their_football_videos/centre_camera.mov")
    capRight = cv2.VideoCapture(os.getcwd() + "/their_football_videos/right_camera.mov")
    # Get total number of frames
    frameCounts = int(capLeft.get(7))
    print frameCounts
    frameAtSpecifiedTime = int((frameCounts/(5*60.0)) * seconds)
    # setting the next frame to be captured to be the frame at
    # the given time
    capLeft.set(cv.CV_CAP_PROP_POS_FRAMES,frameAtSpecifiedTime)
    capCentre.set(cv.CV_CAP_PROP_POS_FRAMES,frameAtSpecifiedTime)
    capRight.set(cv.CV_CAP_PROP_POS_FRAMES,frameAtSpecifiedTime)
    retLeft, leftFrame = capLeft.read()
    retCentre, centreFrame = capCentre.read()
    retRight, rightFrame = capRight.read()
    cv2.imwrite("leftFrame3min.jpg", leftFrame)
    cv2.imwrite("centreFrame3min.jpg", centreFrame)
    cv2.imwrite("rightFrame3min.jpg", rightFrame)
    # setting next frame to be captured back to 1
    capLeft.set(cv.CV_CAP_PROP_POS_FRAMES,1)
    capCentre.set(cv.CV_CAP_PROP_POS_FRAMES,1)
    capRight.set(cv.CV_CAP_PROP_POS_FRAMES,1)
    capLeft.release()
    capCentre.release()
    capRight.release()
    return leftFrame, centreFrame, rightFrame

def histogramEqualization(stitchedImage):
    hist,bins = np.histogram(stitchedImage.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdfMasked = np.ma.masked_equal(cdf,0)
    cdfMasked = (cdfMasked - cdfMasked.min())*255/(cdfMasked.max()-cdfMasked.min())
    cdf = np.ma.filled(cdfMasked,0).astype(np.uint8)
    imgEqualized = cdf[stitchedImage]
    return imgEqualized
 

# Calibrated homography values
# first good left
homographyLeftCentre = np.array([[4.08596402e-01,-1.90538213e-01,1.59402435e+03],[1.20408270e-02,9.67703126e-01,5.57990291e+01],[-3.15819648e-04,2.11969594e-05,1.00000000e+00]])
# second good left
# homographyLeftCentre = np.array([[3.86263441e-01,-2.09435720e-01,1.59375300e+03],[1.08215489e-02,9.54260188e-01,5.62300352e+01],[-3.29649683e-04,1.19431205e-05,1.00000000e+00]])

# homographyLeftCentre = np.matrix(homographyLeftCentre)
# homographyLeftCentreInverse = linalg.inv(homographyLeftCentre)

homographyLeftCentreRight = np.array([[-1.29792062e+00,-1.48633898e-01,7.08595564e+03],[-7.53004129e-02,-1.22009449e+00,5.07314628e+02],[-3.96593862e-04,-1.24223556e-05,1.00000000e+00]])
# homographyLeftCentreRight = np.matrix(homographyLeftCentreRight) 
# homographyLeftCentreRightInverse = linalg.inv(homographyLeftCentreRight)  

capLeft = cv2.VideoCapture(os.getcwd() + "/their_football_videos/left_camera.mp4")
capCentre = cv2.VideoCapture(os.getcwd() + "/their_football_videos/centre_camera.mp4")
capRight = cv2.VideoCapture(os.getcwd() + "/their_football_videos/right_camera.mp4")
# Get total number of frames
# Mac = 7151/mov, PC = 7200/mp4
frameCounts = int(capLeft.get(7))
# Capturing the frame at the 3rd minute of the video
# fps = total frames / 300 seconds
# frame = fps * 180
# set frame to be captured at 4291 (Mac), 4344(Windows)
i = 4320
capLeft.set(cv.CV_CAP_PROP_POS_FRAMES,i)
capCentre.set(cv.CV_CAP_PROP_POS_FRAMES,i)
capRight.set(cv.CV_CAP_PROP_POS_FRAMES,i)
retLeft, left = capLeft.read()
retCentre, centre = capCentre.read()
retRight, right = capRight.read()
# calculating the homography for left and centre frame
# homographyLeftCentre = getHomography(centre,left)
print "homographyLeftCentre is : "
print homographyLeftCentre
# stitching left and centre frame
stitchedLeftCentre,translation1,homoInv1 = stitching(centre,left,homographyLeftCentre)
# calculating the homography for stitched frame and right frame

# homographyLeftCentreRight = getHomography(stitchedLeftCentre,right)
print "homographyLeftCentreRight is "
print homographyLeftCentreRight
stitchedLeftCentreRight,translation2,homoInv2 = stitching(stitchedLeftCentre,right,homographyLeftCentreRight)
cv2.imwrite("leftcentre.jpg", stitchedLeftCentre)
cv2.imwrite("leftcentreright.jpg",stitchedLeftCentreRight)
# Method to cut vertical sides
# getKeypoints(finalImg)
# points=map(itemgetter(0),refPt)
finalImg=stitchedLeftCentreRight[translation1[1,2]+translation2[1,2]:translation1[1,2]+translation2[1,2]+len(centre[:,0]),775:10579]
cv2.imwrite("final.jpg",finalImg)
height, width, channels = finalImg.shape

# Initialising video writer
# High definition codex used
fourcc = cv2.cv.CV_FOURCC('D', 'I', 'V','X')
video = cv2.VideoWriter("stitchedVideo.avi",fourcc,24,(4902,540),True)
capLeft.set(cv.CV_CAP_PROP_POS_FRAMES,0)
capCentre.set(cv.CV_CAP_PROP_POS_FRAMES,0)
capRight.set(cv.CV_CAP_PROP_POS_FRAMES,0)
# 9369(Mac/mov), 9804(Windows/mp4) width
# 1080 (both) height


#stitching video
# for x in range(0, frameCounts):
#     retLeft, left = capLeft.read()
#     retCentre, centre = capCentre.read()
#     retRight, right = capRight.read()
#     combined1,_,_ = stitching(centre,left,homographyLeftCentre)
#     combined2,__,__ = stitching(combined1,right,homographyLeftCentreRight)
#     finalImg = combined2[translation1[1,2]+translation2[1,2]:translation1[1,2]+translation2[1,2]+len(centre[:,0]),775:10579]
#     resize = cv2.resize(finalImg,(4902,540))
#     if x%10 == 0:
#         print x
#     video.write(resize)
# capLeft.release()
# capCentre.release()
# capRight.release()
# video.release()







# # warping
# birdEyeHomography, birdEyeStatus = cv2.findHomography(birdEyeCorners,fieldCorners,0)
# print birdEyeHomography
# newLayout = cv2.warpPerspective()



# # convert to uint16
# # final = np.array(finalImg,dtype = np.uint16)
# # final *= 256
# YCbCrImage = cv2.cvtColor(finalImg, cv2.COLOR_BGR2YCR_CB)
# #cv2.imwrite("ycbcr.jpg", YCbCrImage)
# yImage = YCbCrImage[:,:,0]
# cbImage = YCbCrImage[:,:,1]
# crImage = YCbCrImage[:,:,2]
# correctedYImage = histogramEqualization(yImage)
# correctedCbImage = histogramEqualization(cbImage)
# correctedCrImage = histogramEqualization(crImage)
# correctedImage = np.zeros([height,width,channels])
# correctedImage[:,:,0] = correctedYImage
# correctedImage[:,:,1] = correctedCbImage
# correctedImage[:,:,2] = correctedCrImage
# # correctedImage /= 256
# cv2.imwrite("corrected.jpg", correctedImage)
# correctedImage1 = cv2.cvtColor(correctedImage, cv2.COLOR_YCR_CB2BGR)
# cv2.imwrite("corrected1.jpg", correctedImage1)








