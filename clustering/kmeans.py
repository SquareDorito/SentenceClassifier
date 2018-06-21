import time
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from functools import reduce
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import normalized_mutual_info_score

class KMeans():

    def __init__(self):
        self.currentObjectiveScore=0
        self.totalObjectiveScores=[]
        self.totalNMIScores=[]
        self.totalIterations=[]

        self.cluster_labels=[]

    def getIterations(self):
        return self.totalIterations

    def getObjectiveScores(self):
        return self.totalObjectiveScores

    def assign_clusters(self,data, centers, flag):
        global currentObjectiveScore
        global cluster_labels
        clusters  = {}
        self.currentObjectiveScore=0
        self.cluster_labels=[]
        for x in data:
            if(flag==1):
                bestcenter = min([(i[0], np.linalg.norm(x-centers[i[0]])**2) for i in enumerate(centers)], key=lambda t:t[1])[0]
                self.currentObjectiveScore+=(np.linalg.norm(x-centers[bestcenter])**2)
            else:
                bestcenter = min([(i[0], cdist([x], [centers[i[0]]], metric='cityblock')) for i in enumerate(centers)], key=lambda t:t[1])[0]
                self.currentObjectiveScore+=cdist([x],[centers[bestcenter]], metric='cityblock')[0][0]
            try:
                clusters[bestcenter].append(x)
            except KeyError:
                clusters[bestcenter] = [x]
            self.cluster_labels.append(bestcenter)
        return clusters

    def move_centers(self, centers, clusters, flag):
        newcenters = []
        keys = sorted(clusters.keys())
        for k in keys:
            if(flag==1):
                newcenters.append(np.mean(clusters[k], axis = 0))
            else:
                newcenters.append(np.median(clusters[k], axis = 0))
        return newcenters

    def converged(self,mu, oldmu):
        return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))

    def getFurthest(self,center, data):
        return max([(i[0], np.linalg.norm(center-data[i[0]])) for i in enumerate(data)], key=lambda t:t[1])[0]

    def kmeans(self, data, k, flag, init_flag):
        start_time = time.time()
        # flag=1 for k means, flag=2 for k medians
        self.currentObjectiveScore=0
        self.cluster_labels=[]
        #print(data)
        # init_flag tells us what initialization strategy to use
        oldcenters = data[np.random.choice(data.shape[0], k, replace=False), :]
        if init_flag=='forgy':
            centers = data[np.random.choice(data.shape[0], k, replace=False), :]
        elif init_flag=='maximin':
            temp_data=data
            #get random first center
            centers = data[np.random.choice(data.shape[0], 1, replace=False), :]
            for i in range(1,k):
                index=self.getFurthest(centers[i-1],temp_data)
                #print(index)
                next_center=temp_data[index]
                #remove if selected so can't be selected twice
                temp_data=np.delete(temp_data, index, 0)
                centers = np.vstack([centers, next_center])
        elif init_flag=='k++':
            #stands for k mean/median ++
            #get random first center
            centers = data[np.random.choice(data.shape[0], 1, replace=False), :]
            if flag==1:
                distances = [np.linalg.norm(x-centers[0])**2 for x in data]
                distances_sum=reduce(lambda x,y:x+y, distances)
            else:
                distances = [cdist([x],[centers[0]],metric='cityblock') for x in data]
                distances_sum=reduce(lambda x,y:x+y, distances)
            for _ in range(k-1):
                bestSum=-1
                bestIndex=-1
                for _ in range(len(data)):
                    randVal = random.random()*distances_sum
                    for i in range(len(data)):
                        if randVal <= distances[i]:
                            break
                        else:
                            randVal -= distances[i]
                    if flag==1:
                        tempSum = reduce(lambda x,y:x+y,(min(distances[j], np.linalg.norm(data[j]-data[i])**2) for j in range(len(data))))
                    else:
                        tempSum = reduce(lambda x,y:x+y,(min(distances[j], cdist([data[j]],[data[i]],metric='cityblock')) for j in range(len(data))))
                    if bestSum<0 or tempSum < bestSum:
                        bestSum=tempSum
                        bestIndex=i
                distances_sum = bestSum
                centers = np.vstack([centers, data[bestIndex]])
                if flag==1:
                    distances = [min(distances[i], np.linalg.norm(data[i]-data[bestIndex])**2) for i in range(len(data))]
                else:
                    distances = [min(distances[i], cdist([data[i]],[data[bestIndex]],metric='cityblock')) for i in range(len(data))]
        else:
            print("Not a valid initialization strategy.")
            return
        temp=0
        count=0
        while not self.converged(centers,oldcenters):
            count+=1
            oldcenters = centers
            clusters = self.assign_clusters(data, centers,flag)
            centers = self.move_centers(centers, clusters,flag)
        #print("Number of iterations to converge: "+str(count))
        #print("Clustering Objective O: "+str(objectiveScore))
        self.totalObjectiveScores.append(self.currentObjectiveScore)
        self.totalIterations.append(count)

        print('')
        print('=================== K-Means Statistics ===================')
        print("Number of iterations to converge: "+str(np.mean(self.totalIterations)))
        print("Clustering Objective O: "+str(np.mean(self.totalObjectiveScores)))
        print("Took %s seconds to complete (d=3)." % (time.time() - start_time))
        print('==========================================================')
        print('')
        return(centers, clusters)
