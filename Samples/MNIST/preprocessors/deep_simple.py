from keras.datasets import mnist
from keras.utils import Sequence
from keras.utils import to_categorical
import math

# Simple data preprecsoor to feed data to the neural network
class SimplePreprocessor(Sequence):
    def __init__(self, batchSize):
        self.batchSize = batchSize
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

        # Reshape and normalize data
        self.x_train = self.x_train / 255.0
        self.x_test  = self.x_test / 255.0
        self.x_train = self.x_train.reshape(self.x_train.shape[0], 784)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], 784)

        # Convert y to one hot encoding
        self.y_train = to_categorical(self.y_train, 10)
        self.y_test = to_categorical(self.y_test, 10)


    def __getitem__(self, index):
        startIndex = index*self.batchSize
        try: #Full size batch
            return self.x_train[startIndex : startIndex + self.batchSize], self.y_train[startIndex : startIndex + self.batchSize]
        except IndexError: #Retrieve small batch at the end of the array
            return self.x_train[startIndex:], self.y_train[startIndex:]

    def __len__(self):
        return math.ceil(self.x_train.shape[0]/self.batchSize)


def getPreprocessor():
    return SimplePreprocessor(256)
