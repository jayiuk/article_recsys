import pandas as pd
import numpy as np
import tensorflow
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import sent_tokenize
from tokenizers import tokenizers


tokenizerr = tokenizers
model = Word2Vec()
m = KeyedVectors(vector_size = 5)
data = pd.read_csv('./view_arti.csv.')
word1 = tokenizerr.word_tokenizer(wordpunct_tokenize, data['Title'])
sent1 = tokenizerr.sentence_tokenizer(sent_tokenize, data['Content'], wordpunct_tokenize)
