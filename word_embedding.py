class embedding:
    def __init__(self, model, train_word, word_input, saved_model):
        self.model = model
        self.train_word = train_word
        self.word_input = word_input
        self.saved_model = saved_model

    def make_w2v(model, word_input):
        model_w2v = model(sentences = word_input, vector_size = 100, window = 5, min_count = 5, workers = 4)
        saved_model = model_w2v.save('w2v_model')
        return saved_model
        