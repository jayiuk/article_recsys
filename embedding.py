class embedding:
    def __init__(self, input, embedder):
        self.input = input
        self.embedder = embedder

    def embed(embedder, input):
        w2v = embedder(input)
        return w2v