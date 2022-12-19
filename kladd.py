#%%
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt 
import cv2 as cv   
from ransakelse import *
# from computerVision_explicit import *


img1 = scio.loadmat('./receitaSNS/rgb0001.mat')
img2=scio.loadmat('./receitaSNS/templateSNS.mat')


def findNearestPoint(point, neighbors):#THIS SUCKS, TOO SLOW
    indeks = 0
    dist = np.inf
	
    for i in range(len(neighbors[0,:])): #assumes neighbors is an mxn matrix with n m-dimensional points
        d = np.linalg.norm(point-neighbors[:,i])
		
        if d<dist:
            dist =d
            indeks = i
    return indeks, dist

def matchFeaturepoints(points, template):#returns a list containing the matched template point and its distance to each point
	matchlist=[]
	num_test = points.shape[1]
	num_train = template.shape[1]
	dists = np.zeros((num_test, num_train))
	for j in range(len(points[0,:])):
		dists[j, :] = np.sqrt(np.sum((points.T[j] - template.T) ** 2, axis=1))
		matchlist.append([j,dists[j,:].tolist().index(min(dists[j,:])),min(dists[j,:])])
		print("POINT:",j)

	return np.array(matchlist).transpose()    
# from scipy.io import loadmat''
pts1=img1['d']
pts2=img2['d']

matches=matchFeaturepoints(pts1,pts2)


matchesCopy=np.array([i for i in matches])

inliners=ransac(1000,4,10,matchesCopy,img1,img2)

# plt.plot(matches[2,:])
# plt.show()

