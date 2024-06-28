from tensorflow.keras.preprocessing.text import text_to_word_sequence
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import sent_tokenize
import pandas as pd

def word_tokenizer(tokenizer, input):
    word_input = input.to_string()
    word_token = tokenizer(word_input)
    return word_token

def sentword_tokenizer(sent2wordtokenizer, input):
    sent2word = sent2wordtokenizer(input)
    return sent2word

def sentence_tokenizer(sentokenizer, input, wordtokenizer):
    sentence_input = input.to_string()
    sentence_out = sentokenizer(sentence_input)
    sentence2word_token = [sentword_tokenizer(wordtokenizer, sentence) for sentence in sentence_out]
    return sentence2word_token

def stemming(stemer, input):
    stem_out = [stemer.stem(word) for word in input]
    return stem_out

data = pd.read_csv('./view_arti.csv.')
word1 = word_tokenizer(wordpunct_tokenize, data['Title'])
sent1 = sentence_tokenizer(sent_tokenize, data['Content'], wordpunct_tokenize)
print(word1)
print(sent1)