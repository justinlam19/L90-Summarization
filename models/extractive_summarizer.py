import pickle
import numpy as np
import tqdm

from models.preprocesser import Preprocesser
from nn_utils.activation import ReLU, Sigmoid
from nn_utils.dense import Dense
from nn_utils.layer import Layer
from nn_utils.logistic_regression import LogisticRegression
from nn_utils.loss import binary_cross_entropy, binary_cross_entropy_derivative
from nn_utils.rnn import RNN

class ExtractiveSummarizer:
    def __init__(self):
        self.epochs = 200
        self.batch_size = 64
        self.preprocesser = Preprocesser()
        self.rnn = RNN(
            input_dim=2,
            output_dim=1,
            hidden_dim=64,
        )
        """
        self.network: list[Layer] = [
            Dense(self.preprocesser.vector_size, 1),
            Sigmoid(),
        ]
        self.model = LogisticRegression(input_height=self.preprocesser.vector_size)
        """


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
        articles = [self.preprocesser.article_for_rnn(article) for article in X]

        """
        TODO: Implement me!
        """
        
        lr = 0.02
        decay = 0.95
        for epoch in tqdm.tqdm(range(self.epochs), desc="training", total=self.epochs):
            error = 0
            for i in np.random.permutation(len(articles))[:self.batch_size]:
                article = articles[i]
                y_true = y_true_list[i]
                
                # forward
                """
                x = article
                for layer in self.network:
                    x = layer.forward(x)
                """

                y_pred = self.rnn.sigmoid(
                    np.array([np.squeeze(y) for y in self.rnn.forward(article)])
                )
                
                # error
                error += binary_cross_entropy(y_true, y_pred)

                # backward
                grads = [np.reshape(y, (1, -1)) for y in y_pred - y_true]
                lr *= decay
                self.rnn.backward(grads, learning_rate=lr, momentum=0.9)
            
            with open("f.txt", "a+") as f:
                f.write(f"cost for epoch {epoch+1}: {error / self.batch_size}\n")
        

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
                y_pred = self.rnn.sigmoid(np.array([np.squeeze(y) for y in self.rnn.forward(x)]))
                
                # Randomly assign a score to each sentence. 
                # This is just a placeholder for your actual model.
                sentence_scores = y_pred

            # Pick the top k sentences as summary.
            # Note that this is just one option for choosing sentences.
            top_k_idxs = sorted(range(len(sentence_scores)), key=lambda i: sentence_scores[i], reverse=True)[:k]
            top_sentences = [article[i] for i in top_k_idxs]
            summary = ' . '.join(top_sentences)
            
            yield summary
    