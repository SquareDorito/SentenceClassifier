import numpy as np
from scipy.spatial.distance import cdist
import math
from functools import reduce

currentObjectiveScore=0
totalObjectiveScores=[]
totalNMIScores=[]
totalIterations=[]

cluster_labels=[]

def assign_clusters(data, centers, flag):
    global currentObjectiveScore
    global cluster_labels
    clusters  = {}
    currentObjectiveScore=0
    cluster_labels=[]
    for x in data:
        if(flag==1):
            bestcenter = min([(i[0], np.linalg.norm(x-centers[i[0]])**2) for i in enumerate(centers)], key=lambda t:t[1])[0]
            currentObjectiveScore+=(np.linalg.norm(x-centers[bestcenter])**2)
        else:
            bestcenter = min([(i[0], cdist([x], [centers[i[0]]], metric='cityblock')) for i in enumerate(centers)], key=lambda t:t[1])[0]
            currentObjectiveScore+=cdist([x],[centers[bestcenter]], metric='cityblock')[0][0]
        try:
            clusters[bestcenter].append(x)
        except KeyError:
            clusters[bestcenter] = [x]
        cluster_labels.append(bestcenter)
    return clusters

def move_centers(centers, clusters, flag):
    newcenters = []
    keys = sorted(clusters.keys())
    for k in keys:
        if(flag==1):
            newcenters.append(np.mean(clusters[k], axis = 0))
        else:
            newcenters.append(np.median(clusters[k], axis = 0))
    return newcenters

def converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))

def getFurthest(center, data):
    return max([(i[0], np.linalg.norm(center-data[i[0]])) for i in enumerate(data)], key=lambda t:t[1])[0]

def kmeans(data, k, flag, init_flag):
    # flag=1 for k means, flag=2 for k medians
    global currentObjectiveScore
    global totalObjectiveScores
    currentObjectiveScore=0
    cluster_labels=[]
    # init_flag tells us what initialization strategy to use
    oldcenters = data[np.random.choice(data.shape[0], k, replace=False), :]
    if init_flag=='forgy':
        centers = data[np.random.choice(data.shape[0], k, replace=False), :]
    elif init_flag=='maximin':
        temp_data=data
        #get random first center
        centers = data[np.random.choice(data.shape[0], 1, replace=False), :]
        for i in range(1,k):
            index=getFurthest(centers[i-1],temp_data)
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
    while not converged(centers,oldcenters):
        count+=1
        oldcenters = centers
        clusters = assign_clusters(data, centers,flag)
        centers = move_centers(centers, clusters,flag)
    #print("Number of iterations to converge: "+str(count))
    #print("Clustering Objective O: "+str(objectiveScore))
    totalObjectiveScores.append(currentObjectiveScore)
    totalIterations.append(count)

    for cluster in clusters.keys():
        if(cluster==0):
            plt.scatter(centers[cluster][0],centers[cluster][1],marker='s',color='b')
            clusterX=[]
            clusterY=[]
            for point in clusters[cluster]:
                clusterX.append(point[0])
                clusterY.append(point[1])
            plt.scatter(clusterX,clusterY,color='r')
        elif(cluster==1):
            plt.scatter(centers[cluster][0],centers[cluster][1],marker='s',color='b')
            clusterX=[]
            clusterY=[]
            for point in clusters[cluster]:
                clusterX.append(point[0])
                clusterY.append(point[1])
            plt.scatter(clusterX,clusterY,color='g')
        elif(cluster==2):
            plt.scatter(centers[cluster][0],centers[cluster][1],marker='s',color='b')
            clusterX=[]
            clusterY=[]
            for point in clusters[cluster]:
                clusterX.append(point[0])
                clusterY.append(point[1])
            plt.scatter(clusterX,clusterY,color='c')
        else:
            plt.scatter(centers[cluster][0],centers[cluster][1],marker='s',color='b')
            clusterX=[]
            clusterY=[]
            for point in clusters[cluster]:
                clusterX.append(point[0])
                clusterY.append(point[1])
            plt.scatter(clusterX,clusterY,color='y')
    plt.show()

    return(centers, clusters)

# Data loading code to help you get started
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import normalized_mutual_info_score

arcene_df_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                              + 'arcene/ARCENE/arcene_train.data',
                              delim_whitespace=True,
                              header=None)
arcene_train_labels = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                              + 'arcene/ARCENE/arcene_train.labels',
                              delim_whitespace=True,
                              header=None)

arcene_train_labels_arr=[]

for label in arcene_train_labels.values:
    arcene_train_labels_arr.append(label[0])

# arcene_df_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/'
#                              + 'arcene/ARCENE/arcene_test.data',
#                              delim_whitespace=True,
#                              header=None)
# arcene_df_valid = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/'
#                               + 'arcene/ARCENE/arcene_valid.data',
#                               delim_whitespace=True,
#                               header=None)
# arcene_df = pd.concat([arcene_df_train, arcene_df_test, arcene_df_valid], ignore_index=True)
# arcene_arr = arcene_df.values

print("-------- No PCA -------- (all stats averaged over 100 runs)")
arcene_train = pd.concat([arcene_df_train], ignore_index=True)
arcene_arr_train = arcene_train.values
# start_time = time.time()
# for i in range(1):
#     kmeans(arcene_arr_train,25,1,'forgy')
#     totalNMIScores.append(normalized_mutual_info_score(arcene_train_labels_arr,cluster_labels))
# print("Number of iterations to converge: "+str(np.mean(totalIterations)))
# print("Clustering Objective O: "+str(np.mean(totalObjectiveScores)))
# print("NMI: "+str(np.mean(totalNMIScores)))
# print("Took %s seconds to complete 100 runs." % (time.time() - start_time))
# print("")
#
# totalObjectiveScores=[]
# totalNMIScores=[]
# totalIterations=[]
#
# print("-------- PCA d=100 -------- (all stats averaged over 100 runs)")
# pca = PCA(n_components=100)
# principalComponents = pca.fit_transform(arcene_arr_train)
# start_time = time.time()
# for i in range(1):
#     kmeans(principalComponents,25,1,'forgy')
#     totalNMIScores.append(normalized_mutual_info_score(arcene_train_labels_arr,cluster_labels))
# print("Number of iterations to converge: "+str(np.mean(totalIterations)))
# print("Clustering Objective O: "+str(np.mean(totalObjectiveScores)))
# print("NMI: "+str(np.mean(totalNMIScores)))
# print("Took %s seconds to complete 100 runs." % (time.time() - start_time))
# print("")
#
# totalObjectiveScores=[]
# totalNMIScores=[]
# totalIterations=[]
#
# print("-------- PCA d=10 -------- (all stats averaged over 100 runs)")
# pca = PCA(n_components=10)
# principalComponents = pca.fit_transform(arcene_arr_train)
# start_time = time.time()
# for i in range(1):
#     kmeans(principalComponents,25,1,'forgy')
#     totalNMIScores.append(normalized_mutual_info_score(arcene_train_labels_arr,cluster_labels))
# print("Number of iterations to converge: "+str(np.mean(totalIterations)))
# print("Clustering Objective O: "+str(np.mean(totalObjectiveScores)))
# print("NMI: "+str(np.mean(totalNMIScores)))
# print("Took %s seconds to complete 100 runs." % (time.time() - start_time))
# print("")
#
totalObjectiveScores=[]
totalNMIScores=[]
totalIterations=[]

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(arcene_arr_train)
start_time = time.time()
result=kmeans(principalComponents,25,1,'forgy')
print("Number of iterations to converge: "+str(np.mean(totalIterations)))
print("Clustering Objective O: "+str(np.mean(totalObjectiveScores)))
print("Took %s seconds to complete (d=10)." % (time.time() - start_time))

# anuran_df = pd.read_csv('/Users/kennoh/Downloads/Anuran Calls (MFCCs)/Frogs_MFCCs.csv')
# features = []
# for i in range(22):
#     if i<9:
#         features.append('MFCCs_ '+str(i+1))
#     else:
#         features.append('MFCCs_'+str(i+1))
# anuran_values_arr = anuran_df.loc[:, features].values
# anuran_labels = anuran_df.loc[:,['Species']].values
# anuran_labels_arr=[]
# for label in anuran_labels:
#     anuran_labels_arr.append(label[0])
# #%matplotlib inline
#
# # k_vals=[]
# # i_vals=[]
# # o_vals=[]
# # nmi_vals=[]
# # t_vals=[]
# #
# # for i in range(1,101):
# #     k_vals.append(i)
# #     print("---- kMeans with k=%s ----" % (i))
# #     totalObjectiveScores=[]
# #     totalNMIScores=[]
# #     totalIterations=[]
# #     start_time = time.time()
# #     for j in range(1):
# #         kmeans(anuran_values_arr,i,1,'forgy')
# #         totalNMIScores.append(normalized_mutual_info_score(anuran_labels_arr,cluster_labels))
# #     i_val=np.mean(totalIterations)
# #     print("Number of iterations to converge: "+str(i_val))
# #     i_vals.append(i_val)
# #
# #     o_val=np.mean(totalObjectiveScores)
# #     print("Clustering Objective O: "+str(o_val))
# #     o_vals.append(o_val)
# #
# #     nmi_val=np.mean(totalNMIScores)
# #     print("NMI: "+str(nmi_val))
# #     nmi_vals.append(nmi_val)
# #
# #     t_vals.append(((time.time() - start_time)/1.0))
# #     print("Took %s seconds to complete." % ((time.time() - start_time)/1.0))
# #     print("")

# plt.subplot(2, 2, 1)
# plt.plot(k_vals, i_vals)
# plt.title('Iterations vs. K')
# plt.ylabel('Iterations')
# plt.legend(['Iterations'], loc='upper left')
#
# plt.subplot(2, 2, 2)
# plt.plot(k_vals, t_vals)
# plt.title('Time vs. K')
# plt.ylabel('Time (seconds)')
# plt.legend(['Time'], loc='upper left')
#
# plt.subplot(2, 2, 3)
# plt.plot(k_vals, o_vals)
# plt.title('Clustering Objective vs. K')
# plt.ylabel('Objective (euclidan dist)')
# plt.legend(['Objective'], loc='upper left')
#
# plt.subplot(2, 2, 4)
# plt.plot(k_vals, nmi_vals)
# plt.title('NMI vs. K')
# plt.ylabel('NMI')
# plt.legend(['NMI'], loc='upper left')
# plt.show()

# Note that the last column is a
# sonar_df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/'
#                        + 'undocumented/connectionist-bench/sonar/sonar.all-data',
#                        header=None)
# sonar_arr = sonar_df.values[:, 0:-1] # We remove the label column (rightmost)
# sonar_labels_arr = sonar_df.values[:,-1]
# totalObjectiveScores=[]
# totalNMIScores=[]
# totalIterations=[]
#
# print("-------- kMeans: forgy, k=20, d=10 -------- (all stats averaged over 10 runs)")
# pca = PCA(n_components=10)
# principalComponents = pca.fit_transform(sonar_arr)
# start_time = time.time()
# for i in range(10):
#     kmeans(principalComponents,20,1,'forgy')
#     totalNMIScores.append(normalized_mutual_info_score(sonar_labels_arr,cluster_labels))
# print("Number of iterations to converge: "+str(np.mean(totalIterations)))
# print("Clustering Objective O: "+str(np.mean(totalObjectiveScores)))
# print("NMI: "+str(np.mean(totalNMIScores)))
# print("Took %s seconds to complete per run." % ((time.time() - start_time)/10))
# print("")
#
# totalObjectiveScores=[]
# totalNMIScores=[]
# totalIterations=[]
#
# print("-------- kMeans: maximin, k=20, d=10 -------- (all stats averaged over 10 runs)")
# pca = PCA(n_components=10)
# principalComponents = pca.fit_transform(sonar_arr)
# start_time = time.time()
# for i in range(10):
#     kmeans(principalComponents,20,1,'maximin')
#     totalNMIScores.append(normalized_mutual_info_score(sonar_labels_arr,cluster_labels))
# print("Number of iterations to converge: "+str(np.mean(totalIterations)))
# print("Clustering Objective O: "+str(np.mean(totalObjectiveScores)))
# print("NMI: "+str(np.mean(totalNMIScores)))
# print("Took %s seconds to complete per run." % ((time.time() - start_time)/10))
# print("")
#
# totalObjectiveScores=[]
# totalNMIScores=[]
# totalIterations=[]
#
# print("-------- kMeans: kMeans++, k=20, d=10 -------- (all stats averaged over 10 runs)")
# pca = PCA(n_components=10)
# principalComponents = pca.fit_transform(sonar_arr)
# start_time = time.time()
# for i in range(10):
#     kmeans(principalComponents,20,1,'k++')
#     totalNMIScores.append(normalized_mutual_info_score(sonar_labels_arr,cluster_labels))
# print("Number of iterations to converge: "+str(np.mean(totalIterations)))
# print("Clustering Objective O: "+str(np.mean(totalObjectiveScores)))
# print("NMI: "+str(np.mean(totalNMIScores)))
# print("Took %s seconds to complete per run." % ((time.time() - start_time)/10))
# print("")
#
#
# print("-------- kMedians: forgy, k=20, d=10 -------- (all stats averaged over 10 runs)")
# pca = PCA(n_components=10)
# principalComponents = pca.fit_transform(sonar_arr)
# start_time = time.time()
# for i in range(10):
#     kmeans(principalComponents,20,2,'forgy')
#     totalNMIScores.append(normalized_mutual_info_score(sonar_labels_arr,cluster_labels))
# print("Number of iterations to converge: "+str(np.mean(totalIterations)))
# print("Clustering Objective O: "+str(np.mean(totalObjectiveScores)))
# print("NMI: "+str(np.mean(totalNMIScores)))
# print("Took %s seconds to complete per run." % ((time.time() - start_time)/10))
# print("")
#
# totalObjectiveScores=[]
# totalNMIScores=[]
# totalIterations=[]
#
# print("-------- kMedians: maximin, k=20, d=10 -------- (all stats averaged over 10 runs)")
# start_time = time.time()
# for i in range(10):
#     kmeans(principalComponents,20,2,'maximin')
#     totalNMIScores.append(normalized_mutual_info_score(sonar_labels_arr,cluster_labels))
# print("Number of iterations to converge: "+str(np.mean(totalIterations)))
# print("Clustering Objective O: "+str(np.mean(totalObjectiveScores)))
# print("NMI: "+str(np.mean(totalNMIScores)))
# print("Took %s seconds to complete per run." % ((time.time() - start_time)/10))
# print("")
#
# totalObjectiveScores=[]
# totalNMIScores=[]
# totalIterations=[]
#
# print("-------- kMedians: kMedians++, k=20, d=10 -------- (all stats averaged over 10 runs)")
# start_time = time.time()
# for i in range(10):
#     kmeans(principalComponents,20,2,'k++')
#     totalNMIScores.append(normalized_mutual_info_score(sonar_labels_arr,cluster_labels))
# print("Number of iterations to converge: "+str(np.mean(totalIterations)))
# print("Clustering Objective O: "+str(np.mean(totalObjectiveScores)))
# print("NMI: "+str(np.mean(totalNMIScores)))
# print("Took %s seconds to complete per run." % ((time.time() - start_time)/10))
# print("")
