import os
import numpy as np
import cv2
from matplotlib import pyplot as plt



def basicImageStitching(videoLeft,videoCentre,videoRight,framecounts):
    ret, imageLeft = videoLeft.read()
    ret, imageCentre = videoCentre.read()
    ret, imageRight = videoRight.read()
    grayLeft = cv2.cvtColor(imageLeft, cv2.COLOR_BGR2GRAY)
    grayCentre = cv2.cvtColor(imageCentre, cv2.COLOR_BGR2GRAY)
    grayRight = cv2.cvtColor(imageRight, cv2.COLOR_BGR2GRAY)
    
    surf = cv2.xfeatures2d.SURF_create(5000)
    
    keyPtsLeft, descLeft = surf.detectAndCompute(grayLeft,None)
    keyPtsCentre, descCentre = surf.detectAndCompute(grayCentre,None)
    keyPtsRight, descRight = surf.detectAndCompute(grayRight,None)
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    
    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(descLeft,descCentre,k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in xrange(len(matches))]

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]

    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = 0)

    img3 = cv2.drawMatchesKnn(grayLeft,keyPtsLeft,grayCentre,keyPtsCentre,matches,None,**draw_params)
    plt.imshow(img3,),plt.show()               
                       
capLeft = cv2.VideoCapture(os.getcwd() + "/Football videos/left_camera.mp4")
capCentre = cv2.VideoCapture(os.getcwd() + "/Football videos/centre_camera.mp4")
capRight = cv2.VideoCapture(os.getcwd() + "/Football videos/right_camera.mp4")
frameCounts = int(capLeft.get(7))
basicImageStitching(capLeft,capCentre,capRight,frameCounts)
