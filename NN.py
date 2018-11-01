import numpy as np


class NN(object):
    def __init__(self):
        # set the layer sizes
        self.inputNodes = 8
        self.outputNodes = 8
        self.hiddenNodes = 3

        # initialize weights with random values between 0 and 1
        self.W1 = np.random.randn(self.inputNodes, self.hiddenNodes)
        self.W2 = np.random.randn(self.hiddenNodes, self.outputNodes)

        self.layer1 = None
        self.output = None

    def forward(self, x):
        self.layer1 = self.sigmoid(np.dot(x, self.W1))
        self.output = self.sigmoid(np.dot(self.layer1, self.W2))

    def backward(self, x, y):
        d_values1 = 2*(y - self.output) * self.sigmoid_prime(self.output)
        d_w2 = np.dot(self.layer1.T, d_values1)
        d_values2 = np.dot(d_values1, self.W2.T) * self.sigmoid_prime(self.layer1)
        d_w1 = np.dot(x.T, d_values2)

        # update the weights with the derivative (slope) of the loss function
        self.W1 += d_w1
        self.W2 += d_w2

    def train(self, x, y):
        self.forward(x)
        self.backward(x, y)

    def predict(self, x):
        layer1 = self.sigmoid(np.dot(x, self.W1))
        return self.sigmoid(np.dot(layer1, self.W2))

    def save_weights(self):
        np.savetxt("w1.txt", self.W1, fmt="%s")
        np.savetxt("w2.txt", self.W2, fmt="%s")

    @staticmethod
    def sigmoid(s):
        # activation function
        return 1 / (1 + np.exp(-s))

    @staticmethod
    def sigmoid_prime(s):
        # derivative of sigmoid
        return s * (1 - s)
