import os
import sys
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
from numpy import linalg



def findDimensions(img,homography):
    base_p1 = np.ones(3, np.float32)
    base_p2 = np.ones(3, np.float32)
    base_p3 = np.ones(3, np.float32)
    base_p4 = np.ones(3, np.float32)
    (y, x) = img.shape[:2]
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



def basicImageStitching(videoLeft,videoCentre,videoRight,framecounts):
    
        retLeft, imageLeft = videoLeft.read()
        retCentre, imageCentre = videoCentre.read()
        retRight, imageRight = videoRight.read()           
        grayLeft = cv2.cvtColor(imageLeft, cv2.COLOR_BGR2GRAY)
        grayCentre = cv2.cvtColor(imageCentre, cv2.COLOR_BGR2GRAY)
        grayRight = cv2.cvtColor(imageRight, cv2.COLOR_BGR2GRAY)
        
        surf = cv2.xfeatures2d.SURF_create(9000)
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
             
        # storing good matches as per Lowe's ration test
        goodMatchesLeftCentre = []
        for m,n in matches:
           if m.distance < 0.7*n.distance:
               goodMatchesLeftCentre.append(m)
               
        # finding distance
        sumDistance = 0.0
        for match in goodMatchesLeftCentre:
            sumDistance += match.distance
            
        averagePointDistance = sumDistance/float(len(goodMatchesLeftCentre))
        
        kp1 = []
        kp2 = []
        
        for match in goodMatchesLeftCentre:
            kp2.append(keyPtsCentre[match.trainIdx])
            kp1.append(keyPtsLeft[match.queryIdx])
        
        p1 = np.array([k.pt for k in kp1])
        p2 = np.array([k.pt for k in kp2])
        
        H, status = cv2.findHomography(p1,p2,cv2.RANSAC,5.0)
        inlierRatio = float(np.sum(status))/float(len(status))
        H = H/H[2,2]
        H_inverse = linalg.inv(H)   
        (min_x, min_y, max_x, max_y) = findDimensions(grayLeft, H_inverse)
        
        # Adjust max_x and max_y by base img size
        max_x = max(max_x, grayCentre.shape[1])
        max_y = max(max_y, grayCentre.shape[0])
        move_h = np.matrix(np.identity(3), np.float32)
        if ( min_x < 0 ):
            move_h[0,2] += -min_x
            max_x += -min_x
        if ( min_y < 0 ):
            move_h[1,2] += -min_y
            max_y += -min_y
        mod_inv_h = move_h * H_inverse
        img_w = int(math.ceil(max_x))
        img_h = int(math.ceil(max_y))
        # Warp the new image given the homography from the old image
        imageCentreWarp = cv2.warpPerspective(imageCentre, move_h, (img_w, img_h))
        imageLeftWarp = cv2.warpPerspective(imageLeft,mod_inv_h,(img_w, img_h))
        
        enlarged_base_img = np.zeros((img_h, img_w, 3), np.uint8)
        (ret,data_map) = cv2.threshold(cv2.cvtColor(imageLeftWarp, cv2.COLOR_BGR2GRAY), 
                        0, 255, cv2.THRESH_BINARY)
        enlarged_base_img = cv2.add(enlarged_base_img, imageCentreWarp, 
            mask=np.bitwise_not(data_map), 
            dtype=cv2.CV_8U)
        final_img = cv2.add(enlarged_base_img, imageLeftWarp, 
            dtype=cv2.CV_8U)                        
        return final_img    

               

#capLeft = cv2.VideoCapture(os.getcwd() + "/Football videos/left_camera.mp4")
#capCentre = cv2.VideoCapture(os.getcwd() + "/Football videos/centre_camera.mp4")
#capRight = cv2.VideoCapture(os.getcwd() + "/Football videos/right_camera.mp4")
capLeft = cv2.VideoCapture(os.getcwd() + "/their_football_videos/left_camera.mp4")
capCentre = cv2.VideoCapture(os.getcwd() + "/their_football_videos/centre_camera.mp4")
capRight = cv2.VideoCapture(os.getcwd() + "/their_football_videos/right_camera.mp4")
frameCounts = int(capLeft.get(7))
img = basicImageStitching(capLeft,capCentre,capRight,frameCounts)
cv2.imwrite(os.getcwd() + "/stitch.jpg", img)
capLeft.release()
capCentre.release()
capRight.release()