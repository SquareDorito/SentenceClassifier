#!/usr/bin/python
from InferSent.infersent_embedding import *
from clustering.kmeans import *

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation
import argparse

from sklearn.decomposition import PCA
import skip_thoughts.skipthoughts as skipthoughts

class Annotation3D(Annotation):
    '''Annotate the point xyz with text s'''

    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self,s, xy=(0,0), *args, **kwargs)
        self._verts3d = xyz

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.xy=(xs,ys)
        Annotation.draw(self, renderer)

def annotate3D(ax, s, *args, **kwargs):
    '''add anotation text s to to Axes3d ax'''

    tag = Annotation3D(s, *args, **kwargs)
    ax.add_artist(tag)

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

class SemanticCluster():

    def __init__(self,args):
        self.args=args

        self.sentences=[]
        with open('samples.txt') as f:
            for line in f:
                self.sentences.append(line.strip())

        if self.args.embedding_algo=='infersent':
            self.ie=InfersentEmbedding(500000, 'InferSent/dataset/GloVe/glove.840B.300d.txt',self.sentences)
            self.embeddings=self.ie.infersent_embed()
        elif self.args.embedding_algo=='skip_thoughts':
            self.st_model = skipthoughts.load_model()
            self.st_encoder = skipthoughts.Encoder(self.st_model)
            self.embeddings=self.st_encoder.encode(self.sentences)
        print('Done with embedding...starting clustering.')
        self.principal_components=self.get_nd_data(3)
        self.perform_clustering()
        self.drawGraphs()

    def get_nd_data(self, k):
        pca = PCA(n_components=k)
        principal_components=pca.fit_transform(self.embeddings)

        with open('samples_output_pca3.txt', 'w') as f:
            for pc in principal_components:
                f.write('['+' '.join(str(x) for x in pc)+']')
                f.write('\n')

        return principal_components

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
        colors=generate_colors(int(self.args.k))
        colors=[(x[0]/255.0,x[1]/255.0,x[2]/255.0) for x in colors]

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
        #if args.clustering_algo=='kmeans':
        a=KMeans()
        self.results=a.kmeans(self.principal_components,int(self.args.k),1,'forgy')
        #elif args.clustering_algo=="dbscan":
        #    # TODO: implement DBSCAN if needed
    #        return

def main():
    parser = argparse.ArgumentParser(description='Sentiment Cluster.')
    parser.add_argument('--clustering_algo', nargs='?', default='kmeans', help='Specify whether to use K-Means or DBSCAN clustering algorithm.')
    parser.add_argument('--embedding_algo', nargs='?', default='skip_thoughts', help='Specify sentence embedding technique. Supported techniques: infersent, skip_thoughts.')
    parser.add_argument('--k', nargs='?', default=3,help='Specify the number of clusters (k) for k-means.')
    parser.add_argument('--eps', nargs='?', default=3,help='Maximum distance for two samples to be in same cluster.')
    parser.add_argument('--min_size', nargs='?', default=3,help='Minimum cluster size for DBSCAN.')
    args = parser.parse_args()
    sc=SemanticCluster(args)


if __name__== "__main__":
    main()
