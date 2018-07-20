import skipthoughts
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation
from sklearn.decomposition import PCA

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

model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)

sample_sentences=[]
with open('samples.txt') as f:
    for line in f:
        sample_sentences.append(line.strip())

vectors = encoder.encode(sample_sentences)
with open('samples_output.txt', 'w') as f:
    for e in vectors:
        f.write('['+' '.join(str(x) for x in e)+']')
        f.write('\n')

pca=PCA(n_components=3)
principal_components=pca.fit_transform(vectors)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for j,pc in enumerate(principal_components):
    ax.scatter(pc[0],pc[1],pc[2],c='b',marker='o')
    #ax.annotate('(%s,' %i, xy=(i,j))
    annotate3D(ax, s=str(j+1), xyz=(pc[0],pc[1],pc[2]), fontsize=10, xytext=(-2,2),
               textcoords='offset points', ha='right',va='bottom')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
