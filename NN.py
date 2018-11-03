import numpy as np


class NN(object):
    def __init__(self):
        # set the layer sizes
        self.inputNodes = 8
        self.outputNodes = 8
        self.hiddenNodes = 3

        self.biasInput = 1
        self.biasHidden = 1

        self.learningRate = 0.05

        # initialize weights with random values between 0 and 1
        self.W1 = np.random.randn(self.inputNodes + 1, self.hiddenNodes)
        self.W2 = np.random.randn(self.hiddenNodes + 1, self.outputNodes)

        self.layer1 = None
        self.output = None

    def forward(self, x):
        x = np.insert(x, 0, self.biasInput, 1)  # add bias node input layer
        self.layer1 = self.sigmoid(np.dot(x, self.W1))

        layer1 = np.insert(self.layer1, 0, self.biasHidden, 1)  # add bias node hidden layer
        self.output = self.sigmoid(np.dot(layer1, self.W2))

    def backward(self, x, y):
        layer1 = np.insert(self.layer1, 0, self.biasHidden, 1)  # add bias node hidden layer
        d_values2 = 2 * (y - self.output) * self.sigmoid_prime(self.output)
        d_w2 = np.dot(layer1.T, d_values2)

        x = np.insert(x, 0, self.biasInput, 1)  # add bias node input layer
        d_values1 = np.dot(d_values2, self.W2[1:].T) * self.sigmoid_prime(self.layer1)
        d_w1 = np.dot(x.T, d_values1)

        d_w1 = self.learningRate * d_w1
        d_w2 = self.learningRate * d_w2

        # update the weights with the derivative (slope) of the loss function
        self.W1 += d_w1
        self.W2 += d_w2

    def train(self, x, y):
        self.forward(x)
        self.backward(x, y)

    def predict(self, x):
        x = np.insert(x, 0, self.biasInput, 1)  # add bias node input layer
        layer1 = self.sigmoid(np.dot(x, self.W1))

        layer1 = np.insert(layer1, 0, self.biasHidden, 1)  # add bias node hidden layer
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
