#!/usr/bin/python
from InferSent.infersent_embedding import *
from clustering.kmeans import *

def main():
    ie=InfersentEmbedding(500000, 'InferSent/dataset/GloVe/glove.840B.300d.txt', 'samples.txt')
    embeddings=ie.infersent_embed()

if __name__== "__main__":
    main()
