import numpy as np


class NN(object):
    def __init__(self):
        # set the layer sizes
        self.inputNodes = 8
        self.outputNodes = 8
        self.hiddenNodes = 3

        self.biasInput = 1
        self.biasHidden = 1

        self.learningRate = 0.5

        self.scale_by_input_size = False

        self.reg = 0.0001

        # initialize weights with random values between 0 and 0.1
        self.W1 = np.random.randn(self.inputNodes, self.hiddenNodes) * 0.1
        self.W2 = np.random.randn(self.hiddenNodes, self.outputNodes) * 0.1

        self.bw1 = np.random.randn(1, 3) * 0.1      # bias weight 1
        self.bw2 = np.random.randn(1, 8) * 0.1      # bias weight 2

        self.layer1 = None
        self.output = None

    def forward(self, x):
        # add bias value to input layer
        xb = np.insert(x, 0, 1, 1)

        # weights with bias weight
        W1b = np.insert(self.W1, 0, self.bw1, 0)
        self.layer1 = self.sigmoid(np.dot(xb, W1b))

        # add bias value to input layer
        layer1 = np.insert(self.layer1, 0, 1, 1)

        # weights with bias weight
        W2b = np.insert(self.W2, 0, self.bw2, 0)
        self.output = self.sigmoid(np.dot(layer1, W2b))


    def backward(self, x, y):
        d_values2 = - np.multiply((y - self.output), self.sigmoid_prime(self.output))
        d_w2 = np.dot(self.layer1.T, d_values2)

        d_values1 = np.multiply(np.dot(d_values2, self.W2.T), self.sigmoid_prime(self.layer1))
        d_w1 = np.dot(x.T, d_values1)

        d_b1 = np.average(d_values1, axis=0).reshape(3, 1).T
        d_b2 = np.average(d_values2, axis=0).reshape(8, 1).T

        if self.scale_by_input_size:
            d_w1 = d_w1 / x.shape[0]
            d_w2 = d_w2 / x.shape[0]

            d_b1 = d_b1 / x.shape[0]
            d_b2 = d_b2 / x.shape[0]

        # update the weights with the derivative (slope) of the loss function
        self.W1 -= self.learningRate * (d_w1 + self.reg * self.W1)
        self.W2 -= self.learningRate * (d_w2 + self.reg * self.W2)

        self.bw1 -= self.learningRate * d_b1
        self.bw2 -= self.learningRate * d_b2

    def train(self, x, y):
        self.forward(x)
        self.backward(x, y)

    def predict(self, x):
        # add bias value to input layer
        xb = np.insert(x, 0, 1, 1)

        # weights with bias weight
        W1b = np.insert(self.W1, 0, self.bw1, 0)
        layer1 = self.sigmoid(np.dot(xb, W1b))

        # add bias value to input layer
        layer1 = np.insert(layer1, 0, 1, 1)

        # weights with bias weight
        W2b = np.insert(self.W2, 0, self.bw2, 0)
        return self.sigmoid(np.dot(layer1, W2b))


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
