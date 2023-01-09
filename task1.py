import numpy as np
import sys
from scipy.io import loadmat, savemat
from scipy.spatial.distance import cdist
import cv2 as cv
from matplotlib import pyplot as plt

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
THRESHOLD = 1
NUM_ITERS = 100

class Match:
    pos1: int
    pos2: int
    dist: float

    def __init__(self, pos1, pos2, dist):
        self.pos1 = pos1
        self.pos2 = pos2
        self.dist = dist
    
    def update(self, pos2, dist):
        self.pos2 = pos2
        self.dist = dist

    def __str__(self):
        return f"position 1: {self.pos1}, position 2: {self.pos2}, distance: {self.dist}"


def matching(des1, des2, verbose=True):
    dists = cdist(des1, des2, 'euclidean')
    matches = np.zeros(len(des1), dtype=Match)
    if verbose:
        print(f'Fiding the best match for {len(des1)} descriptors to target {len(des2)} descriptors...')
    for i in range(len(des1)):
        argmin = np.argmin(dists[i])
        matches[i] = Match(i, argmin, dists[i][argmin])
        if verbose:
            print(f'\x1b[2K\r└──> iteration {i + 1} / {len(des1)}', end='')
    if verbose: print('\n')
    return matches
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
                   for c in point_map if dist(c, H) < 5}


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
def drawMatches(image1, image2, point_map, inliers=None, max_points=300):
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
if __name__ == "__main__":
    args = sys.argv
    # template_dir = args[1]
    # input_dir = args[2]
    # output_dir = args[3]

    keyPoints = loadmat('./template/features.mat')
    templateDes = np.array(keyPoints['d']).transpose()
    templateKp = np.array(keyPoints['p']).transpose()
    #templateKp = np.array(list(map(lambda kp :cv.KeyPoint(kp[0], kp[1], 1), templateKp)))

    keyPoints = loadmat('./input/rgb0001.mat')
    rgb0001Des = np.array(keyPoints['d']).transpose()
    rgb0001Kp = np.array(keyPoints['p']).transpose()
    #rgb0001Kp = np.array(list(map(lambda kp :cv.KeyPoint(kp[0], kp[1], 1), rgb0001Kp)))

    #matches = matching(rgb0001Des, templateDes)
    #matches = matching(templateDes, rgb0001Des)
    bf = cv.BFMatcher()
    matches = bf.match(templateDes, rgb0001Des)
    # Apply ratio test
    #pts_cor = np.array([[templateKp[m.pos1][0], templateKp[m.pos1][1], rgb0001Kp[m.pos2][0], rgb0001Kp[m.pos2][1]] for m in matches])
    pts_cor = np.array([[templateKp[m.queryIdx][0], templateKp[m.queryIdx][1], rgb0001Kp[m.trainIdx][0], rgb0001Kp[m.trainIdx][1]] for m in matches])
    

    H, inliers = RANSAC(pts_cor)
    savemat('./output/H_0001.mat', {'H':H})
