class tokenizers:
    def __init__(self, input, tokenizer, sentokenizer, wordtokenizer, stemer, word):
        self.input = input
        self.tokenizer = tokenizer
        self.sentokenizer = sentokenizer
        self.wordtokenizer = wordtokenizer
        self.stemer = stemer
        self.word = word

    def word_tokenizer(tokenizer, input):
        word_input = input.to_string()
        word_token = tokenizer(word_input)
        return word_token

    def sentence_tokenizer(sentokenizer, input, wordtokenizer):
        sentence_input = input.to_string()
        sentence_out = sentokenizer(sentence_input)
        sentence2word_token = [wordtokenizer(sentence) for sentence in sentence_out]
        return sentence2word_token

    def stemming(stemer, input):
        stem_out = [stemer.stem(word) for word in input]
        return stem_out