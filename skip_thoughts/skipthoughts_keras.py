'''
Skip-thought vectors implementation in Keras
Modified from https://github.com/ryankiros/skip-thoughts.

@author: Ken Noh (khnoh@brown.edu)
'''

import os
import numpy as np
import copy
import nltk
from tqdm import tqdm

from collections import OrderedDict, defaultdict
from scipy.linalg import norm
from nltk.tokenize import word_tokenize

profile=False

import keras
from keras.layers import Input, Embedding, GRU, LSTM, Dense, Bidirectional, BatchNormalization
from keras.models import Sequential, Model, load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers

keras.__version__

class SkipThoughts():

    def __init__(self, vocab_k, maxlen, sentences):
        self.sentences=sentences
        self.vocab_k=vocab_k
        self.maxlen=maxlen
        self.embed_dim=150
        self.latent_dim=128
        self.batch_size=64

    def preprocess_text(self):
        tokenizer = Tokenizer(num_words=self.vocab_k)
        tokenizer.fit_on_texts(self.sentences)
        seqs=tokenizer.texts_to_sequences(self.sentences)
        padded_seqs=pad_sequences(seqs,self.maxlen)
        self.x_skip = []
        self.y_before = []
        self.y_after = []
        for i in tqdm(range(1,len(seqs)-1)):
            if len(seqs[i])>4:
                self.x_skip.append(padded_seqs[i].tolist())
                self.y_before.append(padded_seqs[i-1].tolist())
                self.y_after.append(padded_seqs[i+1].tolist())
        self.x_before = np.matrix([[0]+i[:-1] for i in self.y_before])
        self.x_after =np.matrix([[0]+i[:-1] for i in self.y_after])
        self.x_skip = np.matrix(self.x_skip)
        self.y_before = np.matrix(self.y_before)
        self.y_after = np.matrix(self.y_after)

    def build_encoder(self):

        self.encoder_inputs = Input(shape=(self.maxlen,), name='Encoder-Input')
        self.emb_layer = Embedding(self.vocab_k, self.embed_dim, input_length=self.maxlen, name='Body-Word-Embedding', mask_zero=False)
        x=self.emb_layer(self.encoder_inputs)
        _, state_h = GRU(self.latent_dim, return_state=True, name='Encoder-Last-GRU')(x)
        self.encoder_model = Model(inputs=self.encoder_inputs, outputs=state_h, name='Encoder-Model')
        self.seq2seq_encoder_out = self.encoder_model(self.encoder_inputs)

    def build_decoder(self):
        self.decoder_inputs_before = Input(shape=(None,), name='Decoder-Input-before')  # for teacher forcing
        dec_emb_before = self.emb_layer(self.decoder_inputs_before)
        decoder_gru_before = GRU(self.latent_dim, return_state=True, return_sequences=True, name='Decoder-GRU-before')
        decoder_gru_output_before, _ = decoder_gru_before(dec_emb_before, initial_state=self.seq2seq_encoder_out)
        decoder_dense_before = Dense(self.vocab_k, activation='softmax', name='Final-Output-Dense-before')
        self.decoder_outputs_before = decoder_dense_before(decoder_gru_output_before)
        self.decoder_inputs_after = Input(shape=(None,), name='Decoder-Input-after')  # for teacher forcing
        dec_emb_after = self.emb_layer(self.decoder_inputs_after)
        decoder_gru_after = GRU(self.latent_dim, return_state=True, return_sequences=True, name='Decoder-GRU-after')
        decoder_gru_output_after, _ = decoder_gru_after(dec_emb_after, initial_state=self.seq2seq_encoder_out)
        decoder_dense_after = Dense(self.vocab_k, activation='softmax', name='Final-Output-Dense-after')
        self.decoder_outputs_after = decoder_dense_after(decoder_gru_output_after)

    def build_seq2seq(self):
        seq2seq_Model = Model([self.encoder_inputs, self.decoder_inputs_before,self.decoder_inputs_after], [self.decoder_outputs_before,self.decoder_outputs_after])
        seq2seq_Model.compile(optimizer=optimizers.Nadam(lr=0.001), loss='sparse_categorical_crossentropy')
        seq2seq_Model.summary()
        history = seq2seq_Model.fit([self.x_skip,self.x_before, self.x_after], [np.expand_dims(self.y_before, -1),np.expand_dims(self.y_after, -1)],
                  batch_size=self.batch_size,
                  epochs=10,
                  validation_split=0.3)
        seq2seq_Model.save('my_model.h5')

#-----------------------------------------------------------------------------#
# Specify model and table locations here
#-----------------------------------------------------------------------------#
path_to_models = 'data/'
path_to_tables = 'data/'
path_to_umodel = path_to_models + 'uni_skip.npz'
path_to_bmodel = path_to_models + 'bi_skip.npz'

a=SkipThoughts(500000,100,[])
a.preprocess_text()
a.build_encoder()
a.build_decoder()
a.build_seq2seq()

# #Feature extraction
# headlines = tokenizer.texts_to_sequences(data['headline'].values)
# headlines = pad_sequences(headlines,maxlen=maxlen)x = encoder_model.predict(headlines)
# #classifier
# X_train,y_train,X_test,y_test = x[msk],y[msk],x[~msk],y[~msk]
# lr = LogisticRegression().fit(X_train,y_train)
# lr.score(X_test,y_test)
