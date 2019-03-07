from keras.layers import Conv2D, Reshape, Dense
from keras.models import Model

def getModel():
    # This returns a tensor
    inputs = Input(shape=(784,))

    # a layer instance is callable on a tensor, and returns a tensor
    l1 = Conv2D(64, [2,2], strides = [2,2], activation='relu')(inputs)
    l2 = Conv2D(32, [2,2], activation='relu')(l1)
    flat = Reshape((5408,))(l2)

    predictions = Dense(10, activation='softmax')(flat)

    # This creates a model that includes
    # the Input layer and three Dense layers
    return Model(inputs=inputs, outputs=predictions)