import pandas as pd
import numpy as np
import tensorflow
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import sent_tokenize
from tokenizers import tokenizers


tokenizerr = tokenizers
m = KeyedVectors(vector_size = 5)
data = pd.read_csv('./view_arti.csv.')
word1 = tokenizerr.word_tokenizer(wordpunct_tokenize, data['Title'])
sent1 = tokenizerr.sentence_tokenizer(sent_tokenize, data['Content'], wordpunct_tokenize)
word_sent = word1 + sent1
out = Word2Vec(sentences = word_sent, vector_size = 100, window = 5, min_count = 5, workers = 4)
print(type(out))