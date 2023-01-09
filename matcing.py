import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import cdist
import cv2 as cv
from matplotlib import pyplot as plt


BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)


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

if __name__ == "__main__":
    keyPoints = loadmat('./receitaSNS/templateSNS.mat')

    templateDes = np.array(keyPoints['d']).transpose().astype(np.float64)
    templateKp = np.array(keyPoints['p']).transpose().astype(np.float64)

    keyPoints = loadmat('./receitaSNS/rgb0001.mat')

    rgb0001Des = np.array(keyPoints['d']).transpose().astype(np.float64)
    rgb0001Kp = np.array(keyPoints['p']).transpose().astype(np.float64)

    matches = matching(templateDes, rgb0001Des)
    matches = matches.tolist()
    matches.sort(key=lambda x: x.dist)
    best = matches[0:50]

    pts_cor = np.array([[templateKp[m.pos1][0], templateKp[m.pos1][1], rgb0001Kp[m.pos2][0], rgb0001Kp[m.pos2][1]] for m in best])
    img1 = cv.imread('./receitaSNS/templateSNS.jpg',0)          # queryImage
    img2 = cv.imread('./receitaSNS/rgb0001.jpg',0)              # trainImage
    img = drawMatches(img1, img2, pts_cor)
    plt.imshow(img, 'gray')
    plt.show()


            

