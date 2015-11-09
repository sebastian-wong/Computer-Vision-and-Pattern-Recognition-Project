import os, sys
import cv2, numpy as np

LENGTH = 1200
HEIGHT = 800
INPUT_FILE_NAME = "stitched_bg.jpg"
OUTPUT_FILE_NAME = "bird_eye_view_bg.jpg"

beforeWarpImage = cv2.imread(INPUT_FILE_NAME)

#create the field dimension with the height
field = np.zeros([LENGTH,HEIGHT])
TopLeftPointBefore = (float(0), float(0))
TopRightPointBefore = (float(LENGTH), float(0))
BottomRightPointBefore = (float(LENGTH), float(HEIGHT))
BottomLeftPointBefore = (float(0), float(HEIGHT))

#hand picked points
TopLeftPointAfter = (float(2873),float(0))
TopRightPointAfter  = (float(5289), float(0))
BottomRightPointAfter  = (float(8653),float(807))
BottomLeftPointAfter  = (float(0), float(800))

fieldCoordsB = [TopLeftPointBefore, TopRightPointBefore, BottomRightPointBefore, BottomLeftPointBefore]
fieldCoordsA = [TopLeftPointAfter , TopRightPointAfter, BottomRightPointAfter, BottomLeftPointAfter]

BEHomographyMatrix, status = cv2.findHomography(np.array(fieldCoordsA), np.array(fieldCoordsB), 0)
afterWarpImage = cv2.warpPerspective(beforeWarpImage, BEHomographyMatrix , field.shape)

cv2.imwrite(OUTPUT_FILE_NAME, afterWarpImage)
