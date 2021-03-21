
''' module to build simple neural networks '''

import numpy as np
import copy
import math


class Model:
    def __init__(self, *layers):
        self.layers = layers
        self.loss = Crossentropy()

    def __call__(self, optimizer, loss=None):
        self.loss = loss if loss else self.loss
        for layer in self.layers:
            layer(optimizer)
        return self

    def train(self, x, y):
        prediction = self.__feedforward(x)
        gradient = self.loss(y, prediction)
        self.__backpropogation(gradient)
        return prediction

    def predict(self, x):
        return self.__feedforward(x)

    def __feedforward(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def __backpropogation(self, gradient):

        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)


class Layer:
    def __init__(self):
        self.neurons = None
        self.last_data = None
        self.trainable = True

    def __call__(self, optimizer):
        pass

    def forward(self, data):
        pass

    def backward(self, gradient):
        pass


class Dense(Layer):
    def __init__(self, neurons, trainable=True):
        self.neurons = neurons
        self.trainable = trainable

        self.weights = []
        self.bias = []

    def __call__(self, optimizer):
        self.weights_optimizer = copy.copy(optimizer)
        self.bias_optimizer = copy.copy(optimizer)

    def forward(self, data):
        if not len(self.weights):
            limit = 1 / math.sqrt(data.shape[1])
            self.weights  = np.random.uniform(-limit, limit, (data.shape[1], self.neurons))
            self.bias = np.zeros((1, self.neurons))

        self.last_data = data

        output = np.dot(data, self.weights) + self.bias
        return output

    def backward(self, gradient):
        next_gradient = np.dot(gradient, self.weights.T)

        if self.trainable:
            weights_gradient = np.dot(self.last_data.T, gradient)
            bias_gradient = np.sum(gradient, axis=0, keepdims=True)

            self.weights -= self.weights_optimizer.update(weights_gradient)
            self.bias -= self.bias_optimizer.update(bias_gradient)
        
        return next_gradient


class Dropout(Layer):
    def __init__(self, dropout):
        self.dropout = dropout

    def forward(self, data):
        probability = 1.0 - self.dropout

        self.mask = np.random.binomial(size=data.shape, n=1, p=probability)
        data *= self.mask/probability
        return data

    def backward(self, gradient):
        return gradient * self.mask


class Reshape(Layer):
    def __init__(self, shape):
        self.shape = shape
        self.previous_shape = None

    def forward(self, data):
        self.previous_shape = data.shape
        return data.reshape((-1, *self.shape))

    def backward(self, gradient):
        return gradient.reshape(self.previous_shape)


class Flatten(Layer):
    def __init__(self):
        self.previous_shape = None

    def forward(self, data):
        self.previous_shape = data.shape
        return data.reshape((data.shape[0], -1))

    def backward(self, gradient):
        return gradient.reshape(self.previous_shape)


class Activation(Layer):
    def forward(self, data):
        data = self.forward_pass(data)
        self.last_data = data
        return data

    def backward(self, gradient):
        return gradient * self.derivative(self.last_data)

    def derivative(self, x):
        pass


class Sigmoid(Activation):
    def forward_pass(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return x * (1 - x)


class Softmax(Activation):
    def forward_pass(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def derivative(self, x):
        return x * (1 - x)


class ReLU(Activation):
    def forward_pass(self, x):
        return x * (x > 0)

    def derivative(self, x):
        return 1. * (x > 0)


class TanH(Activation):
    def forward_pass(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - np.power(x, 2)


class LeakyReLU(Activation):
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def forward_pass(self, x):
        return np.where(x >= 0, x, self.alpha * x)

    def derivative(self, x):
        return np.where(x >= 0, 1, self.alpha)


class Loss:
    def __init__(self):
        pass

    def acc_score(self, y, p):
        accuracy = np.sum(y == p) / len(y)
        return accuracy


class Crossentropy(Loss): 
    def __call__(self, y, p):
        return (p - y)/(p * (1 - p))

    def loss(self, y, p):
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def acc(self, y, p):
        return self.acc_score(np.argmax(y, axis=1), np.argmax(p, axis=1))


class Optimizer:
    def __init__(self):
        self.learning_rate = None

    def update(self):
        pass


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.epsilon = 1e-8

        self.m = None
        self.v = None

        self.b1 = beta_1
        self.b2 = beta_2

    def update(self, gradient):
        if self.m is None:
            self.m = np.zeros(np.shape(gradient))
            self.v = np.zeros(np.shape(gradient))
        
        self.m = self.b1 * self.m + (1 - self.b1) * gradient
        self.v = self.b2 * self.v + (1 - self.b2) * np.power(gradient, 2)

        m_deriv = self.m / (1 - self.b1)
        v_deriv = self.v / (1 - self.b2)

        weights_update = self.learning_rate * m_deriv / (np.sqrt(v_deriv) + self.epsilon)

        return  weights_update
