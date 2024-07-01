import pandas as pd
import numpy as np
import tensorflow
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import sent_tokenize
from tokenizers import tokenizers

class embedding:
    def __init__(self, input, embedder):
        self.input = input
        self.embedder = embedder

    def embed(embedder, input):
        w2v = embedder.wv(input)
        return w2v

tokenizerr = tokenizers
data = pd.read_csv('./view_arti.csv.')
word1 = tokenizerr.word_tokenizer(wordpunct_tokenize, data['Title'])
sent1 = tokenizerr.sentence_tokenizer(sent_tokenize, data['Content'], wordpunct_tokenize)
word_sent = word1 + sent1
model = Word2Vec(sentences = word_sent, vector_size = 100, window = 5, min_count = 5, workers = 4, sg = 0)
model.save('w2v_model')