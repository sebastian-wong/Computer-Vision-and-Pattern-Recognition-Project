import os
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt



def basicImageStitching(videoLeft,videoCentre,videoRight,framecounts):
    
        retLeft, imageLeft = videoLeft.read()
        retCentre, imageCentre = videoCentre.read()
        retRight, imageRight = videoRight.read()           
        grayLeft = cv2.cvtColor(imageLeft, cv2.COLOR_BGR2GRAY)
        grayCentre = cv2.cvtColor(imageCentre, cv2.COLOR_BGR2GRAY)
        grayRight = cv2.cvtColor(imageRight, cv2.COLOR_BGR2GRAY)

        surf = cv2.xfeatures2d.SURF_create(5000)

        keyPtsLeft, descLeft = surf.detectAndCompute(grayLeft,None)
        keyPtsCentre, descCentre = surf.detectAndCompute(grayCentre,None)
        keyPtsRight, descRight = surf.detectAndCompute(grayRight,None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        # For SIFT and SURF algorithms
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        # Number of times the trees in the index
        # should be recursively traversed
        search_params = dict(checks=50)   
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        # draws k best matches for each keypoint
        matches = flann.knnMatch(descLeft,descCentre,k=2)
        # Need to draw only good matches, so create a mask
        #matchesMask = [[0,0] for i in xrange(len(matches))]

         # for drawing good matching key points
         # ratio test as per Lowe's paper
        # for i,(m,n) in enumerate(matches):
        #     if m.distance < 0.7*n.distance:
        #         matchesMask[i]=[1,0]
        
        # storing good matches as per Lowe's ration test
        goodMatchesLeftCentre = []
        for m,n in matches:
           if m.distance < 0.7*n.distance:
               goodMatchesLeftCentre.append(m)
        
        minimumMatches = 3
        if len(goodMatchesLeftCentre) > minimumMatches:
           srcPts = np.float32([ keyPtsLeft[m.queryIdx].pt for m in goodMatchesLeftCentre ]).reshape(-1,1,2)
           dstPts = np.float32([ keyPtsCentre[m.trainIdx].pt for m in goodMatchesLeftCentre ]).reshape(-1,1,2)
           M, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC,5.0)
           matchesMask = mask.ravel().tolist()
           h,w = grayLeft.shape
           pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
           dst = cv2.perspectiveTransform(pts,M)
           grayCentre = cv2.polylines(grayCentre,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        else:
           print "Insufficient matches"
           matchesMask = None
             
        draw_params = dict(matchColor = (0,255,0),
                         singlePointColor = None,
                         matchesMask = matchesMask,
                         flags = 2)

        # draw_params = dict(matchColor = (0,255,0),
 #                           singlePointColor = (255,0,0),
 #                           matchesMask = matchesMask,
 #                           flags = 0)
        try:
            matchingKeypoints = cv2.drawMatchesKnn(grayLeft,keyPtsLeft,grayCentre,keyPtsCentre,goodMatchesLeftCentre,None,**draw_params)
            #cv2.imwrite(os.getcwd() + '/matchingkeypoints.jpg', matchingKeypoints)
            #plt.imshow(matchingKeypoints,'gray'),plt.show()
        except:
            print "error occurred:" ,sys.exc_info()[0] 
                       
# capLeft = cv2.VideoCapture(os.getcwd() + "/Football videos/left_camera.mp4")
# capCentre = cv2.VideoCapture(os.getcwd() + "/Football videos/centre_camera.mp4")
# capRight = cv2.VideoCapture(os.getcwd() + "/Football videos/right_camera.mp4")
capLeft = cv2.VideoCapture(os.getcwd() + "/their_football_videos/left_camera.mp4")
capCentre = cv2.VideoCapture(os.getcwd() + "/their_football_videos/centre_camera.mp4")
capRight = cv2.VideoCapture(os.getcwd() + "/their_football_videos/right_camera.mp4")
frameCounts = int(capLeft.get(7))
#frameCounts = capLeft.get(cv2.CAP_PROP_FRAME_COUNT)
basicImageStitching(capLeft,capCentre,capRight,frameCounts)
capLeft.release()
capCentre.release()
capRight.release()
