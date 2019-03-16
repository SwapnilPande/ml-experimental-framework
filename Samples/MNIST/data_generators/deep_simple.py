from keras.datasets import mnist
from keras.utils import Sequence
from keras.utils import to_categorical
import math

# Simple data preprecsoor to feed data to the neural network
class SimplePreprocessor(Sequence):
    def __init__(self, batchSize, dataset):
        self.batchSize = batchSize

        if(dataset == "train"): # Train Data
            (self.x, self.y), _ = mnist.load_data()
        else: # Test data
            _ , (self.x, self.y) = mnist.load_data()

        # Reshape and normalize data
        self.x = self.x / 255.0
        self.x = self.x.reshape(self.x.shape[0], 784)

        # Convert y to one hot encoding
        self.y = to_categorical(self.y, 10)


    def __getitem__(self, index):
        startIndex = index*self.batchSize
        try: #Full size batch
            return self.x[startIndex : startIndex + self.batchSize], self.y[startIndex : startIndex + self.batchSize]
        except IndexError: #Retrieve small batch at the end of the array
            return self.x[startIndex:], self.y[startIndex:]

    def __len__(self):
        return math.ceil(self.x.shape[0]/self.batchSize)


def getGenerator(batch_size = 32, dataset = "train"):
    return SimplePreprocessor(batch_size, dataset)
