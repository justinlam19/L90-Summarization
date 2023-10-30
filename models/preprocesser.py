from nltk import word_tokenize
from gensim.models import KeyedVectors
import numpy as np

class Preprocesser:
    def __init__(self, model: str = "models/word2vec-google-news-300.bin"):
        self.model = KeyedVectors.load(model)
        self.model.init_sims(replace=True)
        self.vector_size = self.model.vector_size

    def vectorize_sentence(self, tokens):
        if len(tokens) > 0:
            word_vectors = np.array([self.model[word] for word in tokens if word in self.model.key_to_index])
            if len(word_vectors) > 0:
                return np.reshape(np.mean(word_vectors, axis = 0), (-1, 1))
        return np.reshape(np.zeros(self.vector_size), (-1, 1))

    def normalize(self, a):
        ratio = 2 / (np.max(a) - np.min(a))
        shift = (np.max(a) + np.min(a)) / 2
        return (a - shift) * ratio
    
    def cosine_similarity(self, x, y):
        if not np.any(x) or not np.any(y):
            return 0
        return np.dot(x.ravel(), y.ravel()) / (np.linalg.norm(x) * np.linalg.norm(y))

    def inv_freq(self, tokens):
        if len(tokens) == 0:
            return 0
        n = len(self.model.key_to_index)
        return np.mean([self.model.key_to_index[token] / n for token in tokens])

    def article_for_rnn(self, article: list[str]): 
        tokens_list = [[token for token in word_tokenize(sentence) if token in self.model.key_to_index] for sentence in article]
        
        sentence_vectors = [self.vectorize_sentence(tokens) for tokens in tokens_list]
        mean_vector = np.mean(np.array(sentence_vectors), axis=0)
        cosine_similarities = self.normalize(np.array([self.cosine_similarity(s, mean_vector) for s in sentence_vectors]))
        
        inv_freqs = self.normalize(np.array([self.inv_freq(tokens) for tokens in tokens_list]))
        lengths = self.normalize(np.array([len(tokens) for tokens in tokens_list]))
        positions = np.linspace(-1, 1, num=len(tokens_list))
        
        return [np.reshape(v, (4, -1)) for v in np.vstack((cosine_similarities, inv_freqs, lengths, positions)).T]
        
        """
        tokens_list = [[token for token in word_tokenize(sentence) if token in self.model.key_to_index] for sentence in article]
        sentence_vectors = [self.vectorize_sentence(tokens) for tokens in tokens_list]
        return sentence_vectors
        """

    
    def article_to_matrix(self, article: list[str]):
        mat = []
        for sentence in article:
            tokens = word_tokenize(sentence)
            if len(tokens) > 0:
                word_vectors = np.array([self.model[word] for word in tokens if word in self.model.key_to_index])
                if len(word_vectors) > 0:
                    mat.append(np.mean(word_vectors, axis = 0))
                else:
                    mat.append(np.zeros(self.vector_size))
            else:
                mat.append(np.zeros(self.vector_size))
        return np.array(mat).T
