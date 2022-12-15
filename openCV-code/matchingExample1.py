from scipy.io import loadmat
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

MIN_MATCH_COUNT = 10

template = cv.imread('./openCV-code/templateSNS.jpg')
rgb0001 = cv.imread('./openCV-code/rgb0001.jpg')

keyPoints = loadmat('./openCV-code/templateSNS.mat')

templateDes = np.array(keyPoints['d']).transpose()
templateKp = np.array(keyPoints['p']).transpose()
templateKp = np.array(list(map(lambda kp :cv.KeyPoint(kp[0], kp[1], 1), templateKp)))

keyPoints = loadmat('./openCV-code/rgb0001.mat')

rgb0001Des = np.array(keyPoints['d']).transpose()
rgb0001Kp = np.array(keyPoints['p']).transpose()
rgb0001Kp = np.array(list(map(lambda kp :cv.KeyPoint(kp[0], kp[1], 1), rgb0001Kp)))

bf = cv.BFMatcher()
matches = bf.knnMatch(templateDes, rgb0001Des, k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append([m])
        
img = cv.drawMatchesKnn(template, templateKp, rgb0001, rgb0001Kp,
 good[:15],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img),plt.show()




