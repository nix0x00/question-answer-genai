import numpy as np


class Embeddings:
    def __init__(
        self,
        path: str = "./glove.2024.11b.100d.txt",
    ):
        self.glove = self.load_glove(path)

    def generate_embedding(self, word2idx: dict | None = None):
        if word2idx is None:
            raise ValueError("word2idx must be provided")

        found = 0
        embedding_dim = next(iter(self.glove.values())).shape[0]

        embedding_matrix = np.random.normal(
            scale=0.6, size=(len(word2idx), embedding_dim)
        ).astype(np.float32)

        for word, idx in word2idx.items():
            if word in self.glove:
                embedding_matrix[idx] = self.glove[word]
                found += 1

        # PAD should always be zeros
        embedding_matrix[word2idx["<PAD>"]] = np.zeros(embedding_dim)

        print(f"Found {found}/{len(word2idx)} words in GloVe")

        return embedding_matrix

    def load_glove(self, path: str):
        glove = {}

        with open(path, "r", encoding="utf8") as f:
            for line in f:
                values = line.rstrip().split(" ")
                word = values[0]
                vector = np.asarray(values[1:], dtype=np.float32)
                glove[word] = vector

        return glove
