#!/usr/bin/python
from InferSent.infersent_embedding import *
from clustering.kmeans import *

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation

def generate_colors(n):
    ret = []
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    step = 256 / n
    for i in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        ret.append((r,g,b))
    return ret

class SentimentCluster():

    def __init__(self,k):
        self.k=k
        self.ie=InfersentEmbedding(500000, 'InferSent/dataset/GloVe/glove.840B.300d.txt', 'samples.txt')
        self.embeddings=self.ie.infersent_embed()
        self.principal_components=self.ie.get_3d_data()
        self.perform_clustering()
        self.drawGraphs()

    def drawGraphs(self):
        # ==============
        # First subplot
        # ==============
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        for j,pc in enumerate(self.principal_components):
            ax.scatter(pc[0],pc[1],pc[2],c='b',marker='o')
            #ax.annotate('(%s,' %i, xy=(i,j))
            annotate3D(ax, s=str(j+1), xyz=(pc[0],pc[1],pc[2]), fontsize=10, xytext=(-2,2),
                       textcoords='offset points', ha='right',va='bottom')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        # ==============
        # Second subplot
        # ==============
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        clusters=self.results[1]
        centers=self.results[0]
        colors=generate_colors(self.k)
        colors=[(x[0]/255.0,x[1]/255.0,x[2]/255.0) for x in colors]
        #print(colors)
        for i,cluster in enumerate(clusters.keys()):
            ax.scatter(centers[cluster][0],centers[cluster][1],centers[cluster][2],marker='^',color=colors[i])
            annotate3D(ax, s='Center '+str(i+1), xyz=(centers[cluster][0],centers[cluster][1],centers[cluster][2]), fontsize=10, xytext=(-2,2),
                       textcoords='offset points', ha='right',va='bottom')
            for j,point in enumerate(clusters[cluster]):
                ax.scatter(point[0],point[1],point[2],c=colors[i],marker='o')
                annotate3D(ax, s=str(j+1), xyz=(point[0],point[1],point[2]), fontsize=10, xytext=(-2,2),
                           textcoords='offset points', ha='right',va='bottom')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()

    def perform_clustering(self):
        a=KMeans()
        self.results=a.kmeans(self.principal_components,self.k,1,'forgy')

def main():
    sc=SentimentCluster(3)

if __name__== "__main__":
    main()
