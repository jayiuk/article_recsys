class encoding:
    def __init__(self, input, encoder):
        self.input = input
        self.encoder = encoder

    def vocab(input):
        vocab = {}
        word_list = []
        for word in input:
            word = word.lower()
            word_list.append(word)
            if word not in vocab:
                vocab[word] = 0
            vocab[word] += 1
        return vocab
    