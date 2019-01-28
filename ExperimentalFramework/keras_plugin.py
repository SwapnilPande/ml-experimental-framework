import os
import Experiment

def generateTrainingFiles(label, expInstances):
    os.mkdir(label)
    for instance in expInstances:
        filenameBase = label + "/" + label + "-" + str(instance.instanceIdx)
        with open(filenameBase + "-train" + ".py", "w+") as f:
            f.write("from keras.models import Model\n")
            f.write("from keras.optimizers import SGD\n")
            f.write("import " + instance.model + "\n\n")
            f.write("model = " + instance.model + ".init()\n")
            f.write("model.compile('sgd', loss=['mse'])")



            

