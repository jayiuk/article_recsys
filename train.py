import torch
import torch.nn as nn
from gensim.models import Word2Vec
import pandas as pd
from gensim.models import KeyedVectors
from tokenizers import tokenizers
from model import CNNlstm
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = pd.read_csv('./view_arti.csv.')
word_data = tokenizers.word_tokenizer(wordpunct_tokenize, data[['userID_x', 'articleID', 'Title']])
sent_data = tokenizers.sentence_tokenizer(sent_tokenize, data['Content'], wordpunct_tokenize)
word_sent = word_data + sent_data
data_fin = Word2Vec(sentences = word_sent, vector_size = 100, window = 5, min_count = 5, workers = 4)
learning_rate = 0.002
loss_fn = nn.CrossEntropyLoss()
num_epochs = 100
model = CNNlstm(data_fin).to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)


for i in range(num_epochs):
    print(f"training epochs : {i}")
    optimizer.zero_grad()
    input = data_fin
    out = model.forward(input)
    loss = loss_fn(out)
    loss.backward()
    optimizer.step()
    print(f"training loss : {loss}")