import numpy as np

class ELMClassifier:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        self.activation = self._sigmoid
        self.W = np.random.randn(input_size, hidden_size)
        self.b = np.random.randn(hidden_size)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        H = self.activation(np.dot(X, self.W) + self.b)
        self.beta = np.dot(np.linalg.pinv(H), y.reshape(-1,1))

    def ELMPredict(self, X):
        H = self.activation(np.dot(X, self.W) + self.b)
        Y = np.dot(H, self.beta).ravel()
        return (Y > 0.5).astype(int)
