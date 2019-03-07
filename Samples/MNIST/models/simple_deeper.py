from keras.layers import Input, Dense
from keras.models import Model

def getModel():
    # This returns a tensor
    inputs = Input(shape=(28,28))

    # a layer instance is callable on a tensor, and returns a tensor
    l1 = Dense(64, activation='relu')(inputs)
    l2 = Dense(64, activation='relu')(l1)
    l3 = Dense(32,  activation='relu')(l2)
    l4 = Dense(16,  activation='relu')(l3)

    predictions = Dense(10, activation='softmax')(l4)

    # This creates a model that includes
    # the Input layer and three Dense layers
    return Model(inputs=inputs, outputs=predictions)