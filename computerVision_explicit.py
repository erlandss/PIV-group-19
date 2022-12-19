
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
THRESHOLD = 0.9
NUM_ITERS = 1000
#
#
#
#
#
#
#
def Homography(pts_cor):

    A = []
    for x1, y1, x2, y2 in pts_cor:
        A.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])
        A.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])
    A = np.array(A)

    # Singular Value Decomposition (SVD)
    U, S, V = np.linalg.svd(A)

    # V has shape (9, 9) for any number of input pairs. V[-1] is the eigenvector
    # of (A^T)A with the smallest eigenvalue. Reshape into 3x3 matrix.
    H = np.reshape(V[-1], (3, 3))

    # Normalization
    H = (1 / H.item(8)) * H
    return H
#
#
#
#
#
#
#
#
#
def RANSAC(point_map, threshold=THRESHOLD, verbose=True):
    """
            point_map (List[List[List]]): Map of (x, y) points from one image to the

    """
    if verbose:
        print(f'Running RANSAC with {len(point_map)} points...')
    bestInliers = set()
    homography = None
    for i in range(NUM_ITERS):
        # randomly choose 4 points from the matrix to compute the homography
        pairs = [point_map[i] for i in np.random.choice(len(point_map), 4)]

        H = Homography(pairs)


        inliers = {(c[0], c[1], c[2], c[3])
                   for c in point_map if dist(c, H) < 500}


        if verbose:
            print(f'\x1b[2K\r└──> iteration {i + 1}/{NUM_ITERS} ' +
                  f'\t{len(inliers)} inlier' + ('s ' if len(inliers) != 1 else ' ') +
                  f'\tbest: {len(bestInliers)}', end='')

        if len(inliers) > len(bestInliers):
            bestInliers = inliers
            homography = H
            if len(bestInliers) > (len(point_map) * threshold):
                break

    if verbose:
        print(f'\nNum matches: {len(point_map)}')
        print(f'Num inliers: {len(bestInliers)}')
        print(f'Min inliers: {len(point_map) * threshold}')

    return homography, bestInliers



#
#
#
#
#
#
#
def dist(point_map, H):

    # points in homogeneous coordinates
    p1 = np.array([point_map[0], point_map[1], 1])

    p2 = np.array([point_map[2], point_map[3], 1])

    p2_estimate = np.dot(H, np.transpose(p1))

    p2_estimate = (1 / p2_estimate[2]) * p2_estimate

    return np.linalg.norm(np.transpose(p2) - p2_estimate)
#
#
#
#

#
#
#
#
#
def drawMatches(image1, image2, point_map, inliers=None, max_points=50):
    """
    inliers: set of (x1, y1) points
    """
    rows1, cols1 = image1.shape
    rows2, cols2 = image2.shape

    matchImage = np.zeros((max(rows1, rows2), cols1 + cols2, 3), dtype='uint8')
    matchImage[:rows1, :cols1, :] = np.dstack([image1] * 3)
    matchImage[:rows2, cols1:cols1 + cols2, :] = np.dstack([image2] * 3)

    small_point_map = [point_map[i] for i in np.random.choice(len(point_map), max_points)]

    # draw lines
    for x1, y1, x2, y2 in small_point_map:
        point1 = (int(x1), int(y1))
        point2 = (int(x2 + image1.shape[1]), int(y2))
        color = BLUE if inliers is None else (
            GREEN if (x1, y1, x2, y2) in inliers else RED)

        cv.line(matchImage, point1, point2, color, 1)

    # Draw circles on top of the lines
    for x1, y1, x2, y2 in small_point_map:
        point1 = (int(x1), int(y1))
        point2 = (int(x2 + image1.shape[1]), int(y2))
        cv.circle(matchImage, point1, 5, BLUE, 1)
        cv.circle(matchImage, point2, 5, BLUE, 1)

    return matchImage
#
#
#

#
#

img1 = cv.imread('./receitaSNS/templateSNS.jpg',0)          # queryImage
img2 = cv.imread('./receitaSNS/rgb0001.jpg',0)              # trainImage
# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
matches = cv.BFMatcher(cv.NORM_L2, True).match(des1, des2)

pts_cor = np.array([[kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1], kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]] for m in matches])
src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

H, inliers = RANSAC(pts_cor)

matchesMask = []
outlier = 1
#
#
#

#
#
#
jo = 1
""" __________________________ This for loop is used to get the equivalent variable of matches in cv.findHomography 
    __________________________ But a drawing method has been implemented that requires inliers instead of matches 
for cord in pts_cor:
    outlier = 1
    cord_i = cord.astype(int)

    for eq in inliers:
        eq = np.asarray(eq)
        eq_i = eq.astype(int)

        if (np.array_equal(eq_i, cord_i)):
            outlier = 0
            break



    matchesMask.append(outlier)

"""
h,w = img1.shape
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv.perspectiveTransform(pts,H)
img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)

img3 = drawMatches(img1, img2, pts_cor, inliers)
plt.imshow(img3, 'gray')
plt.show()