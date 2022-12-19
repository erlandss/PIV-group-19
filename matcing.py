import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import cdist


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
        print(f'Fiding the best match for {len(des1)} descriptors to target {len(des2)} descriptors')
    for i in range(len(des1)):
        argmin = np.argmin(dists[i])
        matches[i] = Match(i, argmin, dists[i][argmin])
        if(verbose):
            print(f'\x1b[2K\r└──> iteration {i + 1} / {len(des1)}', end='')
    return matches


keyPoints = loadmat('templateSNS.mat')

templateDes = np.array(keyPoints['d']).transpose()
templateKp = np.array(keyPoints['p']).transpose()

keyPoints = loadmat('rgb0001.mat')

rgb0001Des = np.array(keyPoints['d']).transpose()
rgb0001Kp = np.array(keyPoints['p']).transpose()

matches = matching(templateDes, rgb0001Des)

print('\n' + str(matches[45]))


            

