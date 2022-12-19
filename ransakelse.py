import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt 
import cv2
import random
img1 = scio.loadmat('./receitaSNS/rgb0001.mat')
img2=scio.loadmat('./receitaSNS/rgb0002.mat')


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
            pt=matchingCopy[:,randIndex]

            matchingCopy=matchingCopy.transpose().tolist()

            matchingCopy.pop(randIndex)
            matchingCopy=np.array(matchingCopy).transpose()

            selectedPoints.append(pt)
        selectedPoints=np.array(selectedPoints)

        objectPoints=[]
        templatePoints=[]
        for point in selectedPoints:#Fills the matrices object-/templatePoints with the positions of the feature points in homogeneous coordinates
            objectPoints.append(objectFile['p'][:,int(point[0])])
            templatePoints.append(templateFile['p'][:,int(point[1])])

        objectPoints=np.array(objectPoints)
        templatePoints=np.array(templatePoints)


        #FIND THE HOMOGRAPHY BY SOLVING THE LINEAR EQ Mh=0, WITH h BEING THE (COLUMN)VECTORIZED HOMOGRAPHY
        matrise=[]
        for point in range(n):
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
        # print(np.linalg.det(hvec))
        #EVALUATE THE HOMOGRAPHY

        temp1=objectFile['p'].tolist()
        temp1.append(np.ones(len(objectFile['p'][0,:])).tolist())
        temp1=np.array(temp1)
        temp2=templateFile['p'].tolist()
        temp2.append(np.ones(len(templateFile['p'][0,:])).tolist())
        temp2=np.array(temp2)


        inliners= 0
        for match in matchingCopy.transpose():
            point1 = temp1[:,int(match[0])]
            point2 = temp2[:,int(match[1])]
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

            



# ransac(1,4,2,matchinglist,img1,img2)













