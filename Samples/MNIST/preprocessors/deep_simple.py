from keras.datasets import mnist

# Simple data preprecsoor to feed data to the neural network
class SimplePreprocessor(Sequence):
    def __init__(self, batchSize):
        self.batchSize = batchSize
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

    def __getitem__(self, index):
        startIndex = index*self.batchSize
        try: #Full size batch
            return self.x_train[startIndex : startIndex + self.batchSize], self.y_train[startIndex : startIndex + self.batchSize]
        except IndexError: #Retrieve small batch at the end of the array
            return self.x_train[startIndex:], self.y_train[startIndex:]




def getPreprocessor():
    return SimplePreprocessor(256)