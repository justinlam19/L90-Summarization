import numpy as np
import tqdm

from models.preprocesser import Preprocesser
from nn_utils.loss import binary_cross_entropy
from nn_utils.rnn import RNN

class ExtractiveSummarizer:
    def __init__(self):
        self.epochs = 15
        self.batch_size = 1024
        self.learning_rate = 0.001
        self.decay = 0.05
        self.momentum = 0.9

        self.preprocesser = Preprocesser()
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
        
        split_articles = [[s.strip() for s in x.split('.')] for x in X]
        return split_articles

    def train(self, X: list[list[str]], y: list[list[int]], dummy=False):
        """
        X: list of list of sentences (i.e., comprising an article)
        y: list of yes/no decision for each sentence (as boolean)
        """

        if dummy:
            return

        for article, decisions in tqdm.tqdm(zip(X, y), desc="Validating data shape", total=len(X)):
            assert len(article) == len(decisions), "Article and decisions must have the same length"
        
        y_true_list = [np.array(vector) for vector in y]
        articles = [self.preprocesser.article_for_rnn(article) for article in tqdm.tqdm(X, desc="Preprocessing", total=len(X))]

        """
        TODO: Implement me!
        """
        
        for epoch in tqdm.tqdm(range(self.epochs), desc="training", total=self.epochs):
            error = 0
            self.learning_rate *= (1 - self.decay)

            for i in np.random.choice(len(articles), size=self.batch_size, replace=False):
                article = articles[i]
                y_true = y_true_list[i]
                
                # forward
                """
                x = article
                for layer in self.network:
                    x = layer.forward(x)
                """

                forward_out = np.squeeze(np.array(self.forward_rnn.forward(article)))
                backward_out = np.squeeze(np.array(self.backward_rnn.forward(article[::-1])))[::-1]

                y_pred = self.forward_rnn.sigmoid(np.mean(np.array([forward_out, backward_out]), axis=0))
                
                # error
                error += binary_cross_entropy(y_true, y_pred)

                # backward
                grads = [np.reshape(y / 2, (1, -1)) for y in y_pred - y_true]
                self.forward_rnn.backward(grads, learning_rate=self.learning_rate, momentum=self.momentum)
                self.backward_rnn.backward(grads[::-1], learning_rate=self.learning_rate, momentum=self.momentum)
            
            with open("error.txt", "a+") as f:
                f.write(f"Error for epoch {epoch+1}: {error / self.batch_size}\n")
        

    def predict(self, X: list[list[str]], k=3, dummy=False):
        """
        X: list of list of sentences (i.e., comprising an article)
        """
        
        for article in tqdm.tqdm(X, desc="Running extractive summarizer"):
            """
            TODO: Implement me!
            """

            if dummy:
                sentence_scores = np.random.uniform(size=len(article))
            else:
                x = self.preprocesser.article_for_rnn(article)
                forward_y_pred = self.forward_rnn.sigmoid(np.squeeze(np.array(self.forward_rnn.forward(x))))
                backward_y_pred = self.backward_rnn.sigmoid(np.squeeze(np.array(self.backward_rnn.forward(x[::-1]))))[::-1]
                y_pred = np.mean(np.array([forward_y_pred, backward_y_pred]), axis=0)

                # Randomly assign a score to each sentence.
                # This is just a placeholder for your actual model.
                sentence_scores = y_pred

            # Pick the top k sentences as summary.
            # Note that this is just one option for choosing sentences.
            top_k_idxs = sorted(range(len(sentence_scores)), key=lambda i: sentence_scores[i], reverse=True)[:k]
            top_sentences = [article[i] for i in sorted(top_k_idxs)]
            summary = ' . '.join(top_sentences)
            
            yield summary
    