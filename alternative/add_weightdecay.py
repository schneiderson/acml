import numpy as np


class NN(object):
    def __init__(self):
        # set the layer sizes
        self.inputNodes = 8
        self.outputNodes = 8
        self.hiddenNodes = 3

        self.biasInput = 1.0
        self.biasHidden = 1.0

        self.learningRate = 1
        self.weightDecay = 0.0001

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
        return self.output

    def backward(self, x, y, m):
        layer1 = np.insert(self.layer1, 0, self.biasHidden, 1)  # add bias node hidden layer
#        delta_output = 2 * (y - self.output) * self.sigmoid_prime(self.output) # delta output
        delta_output = self.output - y
        d_w2 = np.dot(layer1.T, delta_output)

        x = np.insert(x, 0, self.biasInput, 1)  # add bias node input layer

        #gPrime = np.multiply(layer1, np.ones((8, 4)) - layer1)
        #delta_hidden = np.multiply(np.dot(theta2.T, delta_output), gPrime)[1:, :]

        delta_hidden = np.dot(delta_output, self.W2[1:].T) * self.sigmoid_prime(self.layer1)
        d_w1 = np.dot(x.T, delta_hidden)

        # update the weights with the derivative (slope) of the loss function
        D1 = (1 / m) * (d_w1 + self.weightDecay * self.get_W1_with_zero_bias())
        D2 = (1 / m) * (d_w2 + self.weightDecay * self.get_W2_with_zero_bias())
        self.W1 = (self.W1 - self.learningRate * D1)
        self.W2 = (self.W2 - self.learningRate * D2)

    def get_W1_with_zero_bias(self):
        copy_w1 = self.W1
        copy_w1[0] = np.zeros((1,3))
        return copy_w1

    def get_W2_with_zero_bias(self):
        copy_w2 = self.W2
        copy_w2[0] = np.zeros((1, 8))
        return copy_w2

    def train(self, x, y, m):
        self.forward(x)
        self.backward(x, y, m)

    def predict(self, x):
        return self.forward(x)


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
