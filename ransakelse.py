import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt 
import cv2
import random
from matcing import *
img1 = scio.loadmat('./receitaSNS/rgb0001.mat')
img2=scio.loadmat('./receitaSNS/templateSNS.mat')


#TODO: make separate function for homography, and for making distance calculations faster

def ransac(k,n,epsilon,matchingList,objectFile,templateFile):
    maxInliners=0
    inlinerList=[]
    for i in range(k):
        currentInliners=[]
        selectedPoints = []

        matchingCopy= np.array([pt for pt in matchingList])
        for j in range(n):#Select n unique points from the copy of the matching list
            randIndex = random.randint(0,max(np.shape(matchingCopy))-1)
            pt=matchingCopy[randIndex]#BARE HENT UT Match OBJECT X

            matchingCopy=matchingCopy.tolist()

            matchingCopy.pop(randIndex)
            matchingCopy=np.array(matchingCopy)

            selectedPoints.append(pt)
        selectedPoints=np.array(selectedPoints)

        # objectPoints=[]
        # templatePoints=[]
        # for point in selectedPoints:#Fills the matrices object-/templatePoints with the positions of the feature points in homogeneous coordinates
        #     objectPoints.append(objectFile['p'][:,point.pos1])#BYTT fra INDEX TIL POS1 OG POS2 X
        #     templatePoints.append(templateFile['p'][:,point.pos2])

        # objectPoints=np.array(objectPoints)
        # templatePoints=np.array(templatePoints)


        # #FIND THE HOMOGRAPHY BY SOLVING THE LINEAR EQ Mh=0, WITH h BEING THE (COLUMN)VECTORIZED HOMOGRAPHY
        # matrise=[]
        # for point in range(n):
        #     x1,y1 = objectPoints[point,0],objectPoints[point,1]
        #     x2,y2 = templatePoints[point,0],templatePoints[point,1]
        #     matrise.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])
        #     matrise.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])

        # matrise = np.array(matrise)

        # kvadratMatrise = np.matmul(matrise.transpose(),matrise)

        # eigs, eigvecs=np.linalg.eig(kvadratMatrise)
        # minindex = eigs.tolist().index(min(eigs))
        # hvec = eigvecs[:,minindex]
        # hvec = hvec.reshape(3,3)
        # # print(np.linalg.det(hvec))
        hvec = homography(selectedPoints,objectFile,templateFile)
        #EVALUATE THE HOMOGRAPHY

        temp1=objectFile['p'].tolist()
        temp1.append(np.ones(len(objectFile['p'][0,:])).tolist())
        temp1=np.array(temp1)
        temp2=templateFile['p'].tolist()
        temp2.append(np.ones(len(templateFile['p'][0,:])).tolist())
        temp2=np.array(temp2)


        inliners= 0
        # print(f'\x1b[2K\r└──> iteration {i + 1} / {k}', end='')
        for match in matchingCopy:
            point1 = temp1[:,match.pos1]#FIX from indexes to pos1/2
            point2 = temp2[:,match.pos2]
            estimatedPoint2 = np.dot(hvec,np.transpose(point1))
            estimatedPoint2 = estimatedPoint2 * (1/estimatedPoint2[2])

            if np.linalg.norm(estimatedPoint2-np.transpose(point2))<epsilon:
                inliners+=1
                currentInliners.append(match)
        if inliners>maxInliners:
            maxInliners=inliners
            inlinerList=currentInliners
    print("Best homography yielded ",maxInliners,"inliners")
    return inlinerList


def homography(pointList, objectFile, templateFile):
        matrise=[]
        objectPoints=[]
        templatePoints=[]
        for point in pointList:#Fills the matrices object-/templatePoints with the positions of the feature points in homogeneous coordinates
            objectPoints.append(objectFile['p'][:,point.pos1])
            templatePoints.append(templateFile['p'][:,point.pos2])

        objectPoints=np.array(objectPoints)
        templatePoints=np.array(templatePoints)
        for point in range(len(pointList)):
            x1,y1 = objectPoints[point,0],objectPoints[point,1]
            x2,y2 = templatePoints[point,0],templatePoints[point,1]
            matrise.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])
            matrise.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])

        matrise = np.array(matrise)

        kvadratMatrise = np.matmul(matrise.transpose(),matrise)

        eigs, eigvecs=np.linalg.eig(kvadratMatrise)
        minindex = eigs.tolist().index(min(eigs))
        hvec = eigvecs[:,minindex]
        hvec = hvec.reshape(3,3)    
        return hvec


keyPoints = scio.loadmat('./receitaSNS/templateSNS.mat')

templateDes = np.array(keyPoints['d']).transpose()
templateKp = np.array(keyPoints['p']).transpose()

keyPoints = scio.loadmat('./receitaSNS/rgb0001.mat')

rgb0001Des = np.array(keyPoints['d']).transpose()
rgb0001Kp = np.array(keyPoints['p']).transpose()

matches = matching(rgb0001Des, templateDes )
            

inliners=ransac(1000,4,10,matches,img1,img2)

homo = homography(inliners,img1,img2)

# ransac(1,4,2,matchinglist,img1,img2)













