#!/usr/bin/python
#from InferSent.infersent_embedding import *
from clustering.kmeans import *

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation
from sklearn.cluster import DBSCAN
from sklearn import metrics
import argparse

from sklearn.decomposition import PCA

# Importing skip_thoughts_theano
import skip_thoughts_theano.skipthoughts as skipthoughts_theano
# Importing skip_thoughts_tf
import skip_thoughts_tf.skip_thoughts as skipthoughts_tf



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

class SemanticClusterModule():

    def __init__(self,args):
        #support for command line args
        self.args=args
        self.sentences=[]
        self.model=None
        self.encoder=None
        self.embeddings=[]
        self.principal_components=[]
        #self.principal_components=self.get_nd_data(3)
        #self.perform_clustering()
        #self.drawGraphs()
        self.algo_name=""

    def load_data(self, file):
        with open(file) as f:
            for line in f:
                self.sentences.append(line.strip())

    def load_model(self, algo_name):
        self.algo_name=algo_name
        if algo_name=='infersent':
            # Outdated, no longer using infersent embedding technique.
            # self.ie=InfersentEmbedding(500000, 'InferSent/dataset/GloVe/glove.840B.300d.txt',self.sentences)
            # self.embeddings=self.ie.infersent_embed()
            pass
        elif algo_name=='skipthoughts_theano':
            self.model = skipthoughts_theano.load_model()
            self.encoder = skipthoughts_theano.Encoder(self.model)
        elif algo_name=='skipthoughts_tf':
            UNI_MODEL_PATH = "skip_thoughts/pretrained/skip_thoughts_uni_2017_02_02/"
            self.encoder=skipthoughts_tf.encoder_manager.EncoderManager()
            self.encoder.load_model(skipthoughts_tf.configuration.model_config(),
                       vocabulary_file=UNI_MODEL_PATH+"vocab.txt",
                       embedding_matrix_file=UNI_MODEL_PATH+"embeddings.npy",
                       checkpoint_path=UNI_MODEL_PATH+"model.ckpt-501424")
        else:
            print("Invalid algo name, please try again with either: skipthoughts_tf, skip_thoughts_theano.")
            return False
        return True

    def encode(self):
        if not self.encoder:
            print("Please load a model before attempting to encode.")
            return None
        self.embeddings=self.encoder.encode(self.sentences)
        return self.embeddings

    def get_nd_data(self, k, output_file=False):
        pca = PCA(n_components=k)
        self.principal_components=pca.fit_transform(self.embeddings)

        if output_file:
            # Print 3D embeddings to file for testing
            with open('samples_output_pca3.txt', 'w') as f:
                for pc in self.principal_components:
                    f.write('['+' '.join(str(x) for x in pc)+']')
                    f.write('\n')

        return self.principal_components

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

    def perform_clustering(self, clustering_algo):
        if clustering_algo=='kmeans':
            a=KMeans()
            self.clustering_results=a.kmeans(self.principal_components,int(self.args.k),1,'forgy')
            return self.clustering_results
        elif clustering_algo=="dbscan":
            db = DBSCAN(eps=0.3, min_samples=3,metric='cosine').fit(self.embeddings)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_
            #print(labels)
            n_clusters_ = len(set(labels)) - (1 if -1 else 0)
            print(n_clusters_)

            self.cluster_dict={}
            for i,l in enumerate(labels):
                try:
                    self.cluster_dict[l].append(self.sentences[i])
                except:
                    self.cluster_dict[l]=[self.sentences[i]]

            with open('cluster_output.txt', 'w') as f:
                for key in self.cluster_dict:
                    if key==-1:
                        continue
                    f.write('============= Cluster '+str(key)+': =============\n')
                    for j,sentence in enumerate(self.cluster_dict[key]):
                        f.write(str(j)+'. '+sentence+'\n')
            return self.cluster_dict

def main():
    parser = argparse.ArgumentParser(description='Sentiment Cluster.')
    parser.add_argument('--clustering_algo', nargs='?', default='kmeans', help='Specify whether to use K-Means or DBSCAN clustering algorithm.')
    parser.add_argument('--embedding_algo', nargs='?', default='skip_thoughts', help='Specify sentence embedding technique. Supported techniques: infersent, skip_thoughts.')
    parser.add_argument('--k', nargs='?', default=3,help='Specify the number of clusters (k) for k-means.')
    parser.add_argument('--eps', nargs='?', default=3,help='Maximum distance for two samples to be in same cluster.')
    parser.add_argument('--min_size', nargs='?', default=3,help='Minimum cluster size for DBSCAN.')
    args = parser.parse_args()
    sc=SemanticClusterModule(args)


if __name__== "__main__":
    main()
