import collections
import numpy as np

import keras
from keras import backend as K

from skip_thoughts import skip_thoughts_encoder

class EncoderManager(object):

    def __init__(self):
        self.encoders=[]
        self.sessions=[]

    def load_model(self,model_config,vocab_file, embedding_matrix_file, checkpoint_path):
        with open(vocab_file,'r') as f:
            lines=list(f.readlines())
        #print(lines)
        reverse_vocab=[line.strip() for line in lines]
        embedding_matrix=np.load(embedding_matrix_file)
        word_embeddings = collections.OrderedDict(zip(reverse_vocab,embedding_matrix))

        encoder = skip_thoughts_encoder.SkipThoughtsEncoder(word_embeddings)
        restore_model = encoder.build_graph_from_config(model_config,
                                                      checkpoint_path)

        tf_sess = K.get_session()
        restore_model(tf_sess)
        
        self.encoders.append(encoder)
        self.sessions.append(tf_sess)

test=EncoderManager()
test.load_model(None,'pretrained/skip_thoughts_uni_2017_02_02/vocab.txt',None,None)
