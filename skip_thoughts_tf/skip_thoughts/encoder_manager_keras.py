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

    def encode(self,
             data,
             use_norm=True,
             verbose=False,
             batch_size=128,
             use_eos=False):
    """Encodes a sequence of sentences as skip-thought vectors.

    Args:
      data: A list of input strings.
      use_norm: If True, normalize output skip-thought vectors to unit L2 norm.
      verbose: Whether to log every batch.
      batch_size: Batch size for the RNN encoders.
      use_eos: If True, append the end-of-sentence word to each input sentence.

    Returns:
      thought_vectors: A list of numpy arrays corresponding to 'data'.

    Raises:
      ValueError: If called before calling load_encoder.
    """
    if not self.encoders:
      raise ValueError(
          "Must call load_model at least once before calling encode.")

    encoded = []
    for encoder, sess in zip(self.encoders, self.sessions):
      encoded.append(
          np.array(
              encoder.encode(
                  sess,
                  data,
                  use_norm=use_norm,
                  verbose=verbose,
                  batch_size=batch_size,
                  use_eos=use_eos)))

    return np.concatenate(encoded, axis=1)

  def close(self):
    """Closes the active TensorFlow & Keras sessions."""
    for sess in self.sessions:
      sess.close()