import os
import Experiment

# Returns the strings to write to import necessary packages
def imports(expInstance):
    outStr = "from keras.models import Model\n" 
    outStr += "from keras.optimizers import SGD\n"
    outStr += "from keras.optimizers import SGD\n"
    outStr += "import" + expInstance.model + "\n\n"
    return outStr

def generateTrainingFiles(label, expInstances):
    os.mkdir(label)
    for instance in expInstances:

        filenameBase = label + "/" + label + "-" + str(instance.instanceIdx)

        with open(filenameBase + "-train" + ".py", "w+") as f:

            # Writes import statements to file
            f.write(imports(instance))



            

