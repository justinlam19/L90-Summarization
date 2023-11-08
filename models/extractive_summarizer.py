import pickle
import numpy as np
import tqdm

from models.preprocesser import Preprocesser
from nn_utils.loss import binary_cross_entropy
from nn_utils.rnn import RNN


class ExtractiveSummarizer:
    def __init__(self, word2vec_model: str, dummy: bool):
        self.dummy = dummy

        if not self.dummy:        
            self.epochs = 15
            self.batch_size = 1024
            self.learning_rate = 0.001
            self.decay = 0.05
            self.momentum = 0.9

            self.preprocesser = Preprocesser(word2vec_model)
            self.forward_rnn = RNN(
                input_dim=4,
                output_dim=1,
                hidden_dim=64,
            )
            self.backward_rnn = RNN(
                input_dim=4,
                output_dim=1,
                hidden_dim=64,
            )

    def preprocess(self, X: list[list[str]]):
        """
        X: list of list of sentences (i.e., comprising an article)
        """
        split_articles = [[s.strip() for s in x.split(".")] for x in X]
        return split_articles

    def train(self, X: list[list[str]], y: list[list[int]], save=None):
        """
        X: list of list of sentences (i.e., comprising an article)
        y: list of yes/no decision for each sentence (as boolean)
        """
        if self.dummy:
            return

        for article, decisions in tqdm.tqdm(
            zip(X, y), desc="Validating data shape", total=len(X)
        ):
            assert len(article) == len(
                decisions
            ), "Article and decisions must have the same length"

        y_true_list = [np.array(vector) for vector in y]
        articles = [
            self.preprocesser.article_for_rnn(article)
            for article in tqdm.tqdm(X, desc="Preprocessing", total=len(X))
        ]

        for epoch in tqdm.tqdm(range(self.epochs), desc="training", total=self.epochs):
            error = 0
            self.learning_rate *= 1 - self.decay

            for i in np.random.choice(
                len(articles), size=self.batch_size, replace=False
            ):
                article = articles[i]
                y_true = y_true_list[i]

                # forward
                forward_out = np.squeeze(np.array(self.forward_rnn.forward(article)))
                backward_out = np.squeeze(
                    np.array(self.backward_rnn.forward(article[::-1]))
                )[::-1]
                y_pred = self.forward_rnn.sigmoid(
                    np.mean(np.array([forward_out, backward_out]), axis=0)
                )

                # error
                error += binary_cross_entropy(y_true, y_pred)

                # backward
                grads = [np.reshape(y / 2, (1, -1)) for y in y_pred - y_true]
                self.forward_rnn.backward(
                    grads, learning_rate=self.learning_rate, momentum=self.momentum
                )
                self.backward_rnn.backward(
                    grads[::-1],
                    learning_rate=self.learning_rate,
                    momentum=self.momentum,
                )

            with open("error.txt", "a+") as f:
                f.write(f"Error for epoch {epoch+1}: {error / self.batch_size}\n")

        if save:
            with open(save, "wb+") as f:
                pickle.dump(
                    {
                        "forward_rnn": self.forward_rnn,
                        "backward_rnn": self.backward_rnn,
                    },
                    f,
                )

    def predict(self, X: list[list[str]], k=3, load=None):
        """
        X: list of list of sentences (i.e., comprising an article)
        """
        if load:
            with open(load, "rb+") as f:
                models = pickle.load(f)
            self.forward_rnn = models["forward_rnn"]
            self.backward_rnn = models["backward_rnn"]

        for article in tqdm.tqdm(X, desc="Running extractive summarizer"):
            if self.dummy:
                sentence_scores = np.random.uniform(size=len(article))
            else:
                x = self.preprocesser.article_for_rnn(article)
                forward_out = np.squeeze(np.array(self.forward_rnn.forward(x)))
                backward_out = np.squeeze(
                    np.array(self.backward_rnn.forward(x[::-1]))
                )[::-1]
                sentence_scores = self.forward_rnn.sigmoid(
                    np.mean(np.array([forward_out, backward_out]), axis=0)
                )

            # Pick the top k sentences as summary.
            top_k_idxs = sorted(
                range(len(sentence_scores)),
                key=lambda i: sentence_scores[i],
                reverse=True,
            )[:k]
            top_sentences = [article[i] for i in sorted(top_k_idxs)]
            summary = " . ".join(top_sentences)

            yield summary
