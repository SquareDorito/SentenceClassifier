from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os.path
import scipy.spatial.distance as sd
from skip_thoughts import configuration
from skip_thoughts import encoder_manager

from sklearn.cluster import DBSCAN
from sklearn import metrics

import os.path

# ======================================================= #
#                Set paths to models here:                #
# ======================================================= #
UNI_MODEL_PATH = "skip_thoughts/pretrained/skip_thoughts_uni_2017_02_02/"
BI_MODEL_PATH = "skip_thoughts/pretrained/skip_thoughts_bi_2017_02_16/"
#VOCAB_FILE = BIDIR_MODEL_PATH + "vocab.txt"
#EMBEDDING_MATRIX_FILE = BIDIR_MODEL_PATH + "embeddings.npy"
#CHECKPOINT_PATH = "/path/to/model.ckpt-9999"
# The following directory should contain files rt-polarity.neg and
# rt-polarity.pos.
MR_DATA_DIR = "rt-polaritydata/rt-polaritydata/"


# ======================================================= #
#                Load the trained models.                 #
# ======================================================= #

# unidirectionally-trained model
encoder = encoder_manager.EncoderManager()
encoder.load_model(configuration.model_config(),
                   vocabulary_file=UNI_MODEL_PATH+"vocab.txt",
                   embedding_matrix_file=UNI_MODEL_PATH+"embeddings.npy",
                   checkpoint_path=UNI_MODEL_PATH+"model.ckpt-501424")

# bidirectionally-trained model
# encoder.load_model(configuration.model_config(bidirectional_encoder=True),
#                    vocabulary_file=BI_MODEL_PATH+"vocab.txt",
#                    embedding_matrix_file=BI_MODEL_PATH+"embeddings.npy",
#                    checkpoint_path=BI_MODEL_PATH+"model.ckpt-500008")

# ======================================================= #
#                Loading in movie dataset.                #
# ======================================================= #
embeddings=[]

if os.path.isfile('sample_output.txt'):
    print('Found output file. Using saved encodings...')
    with open('sample_output.txt', 'r') as f:
        for line in f:
            temp=line.strip('\n').strip('[').strip(']').split(' ')
            temp=[float(x) for x in temp]
            embeddings.append(temp)
else:
    data = []
    with open(os.path.join(MR_DATA_DIR, 'rt-polarity.neg'), 'rb') as f:
      data.extend([line.decode('latin-1').strip() for line in f])
    with open(os.path.join(MR_DATA_DIR, 'rt-polarity.pos'), 'rb') as f:
      data.extend([line.decode('latin-1').strip() for line in f])

    #print(data)
    embeddings = encoder.encode(data)
    print(len(embeddings))
    print(len(embeddings[0]))
    with open('sample_output.txt', 'w') as f:
        for e in embeddings:
            f.write('['+' '.join(str(x) for x in e)+']\n')

print(len(embeddings))
print(len(embeddings[0]))
db = DBSCAN(eps=10, min_samples=3).fit(embeddings)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
print(labels)
n_clusters_ = len(set(labels)) - (1 if -1 else 0)
print(n_clusters_)
