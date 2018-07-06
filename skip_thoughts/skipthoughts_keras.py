'''
Skip-thought vectors implementation in Keras
Modified from https://github.com/ryankiros/skip-thoughts.

@author: Ken Noh (khnoh@brown.edu)
'''

import os
import cPickel as pkl
import numpy as np
import copy
import nltk

from collections import OrderedDict, defaultdict
from scipy.linalg import norm
from nltk.tokenize import word_tokenize

profile=False

import keras
keras.__version__

#-----------------------------------------------------------------------------#
# Specify model and table locations here
#-----------------------------------------------------------------------------#
path_to_models = 'data/'
path_to_tables = 'data/'
path_to_umodel = path_to_models + 'uni_skip.npz'
path_to_bmodel = path_to_models + 'bi_skip.npz'

def load_model():
    '''
    Loads model with saved tabels
    '''
    with open('%s.pkl'%path_to_umodel, 'rb') as f:
        uoptions = pkl.load(f)
    with open('%s.pkl'%path_to_bmodel, 'rb') as f:
        boptions = pkl.load(f)

def init_params(options):
    """
    initialize all parameters needed for the encoder
    """
    params = OrderedDict()

    # embedding
    params['Wemb'] = norm_weight(options['n_words_src'], options['dim_word'])

    # encoder: GRU
    params = get_layer(options['encoder'])[0](options, params, prefix='encoder',
                                              nin=options['dim_word'], dim=options['dim'])
    return params
