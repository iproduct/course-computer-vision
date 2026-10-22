import numpy as np
import argparse
import imutils
import cv2
import math
from directlt import dlt
from ransac import ransac
from matchPoints import matchPoints

img1 = cv2.imread('data/mount_left.png')
img2 = cv2.imread('data/mount_right.png')

img2 = imutils.resize(img2, width=400)
img1 = imutils.resize(img1, width=400)

[pts1 , pts2] = matchPoints(img1, img2)

H = ransac(pts1,pts2, 20, 0.98, 4.0) ## N, t, thresh
H = np.float64([pt for pt in H])

# result = cv2.warpPerspective(img1, H,(img1.shape[1] + img2.shape[1], img1.shape[0]))
width = img1.shape[1] + img2.shape[1]
height = max(img1.shape[0], img2.shape[0])
result = cv2.warpPerspective(img1, H, (width, height))
result[0:img2.shape[0], 0:img2.shape[1]] = img2

cv2.imwrite('output/result.png', result)
