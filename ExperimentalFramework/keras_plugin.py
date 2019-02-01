import os
import Experiment

# Returns the strings to write to import necessary packages
def imports(expInstance):
    outStr = "# Imports"
    outStr += "from keras.models import Model\n"
    #Import the correct optimizer for this model
    outStr += "from keras.optimizers import %(type)s\n" % {"type" : expInstance.optimizer["type"]}
    outStr += "import " + expInstance.model + "\n\n"
    return outStr

def optimizer(expInstance):
    # Convert dictionary of arguments into list of arguments with = sign
    argList = [str(key) + "=" + str(value) for key,value in expInstance.optimizer["optimizer_parameters"].items()]
    #convert into comma separated string
    argString = ",".join(argList)
    # Generate the optimizer object with the correct parameters
    outStr = "# Define optimizer\n"
    outStr += "optimizer = %(type)s(%(args)s)\n\n" % {
        "type" : expInstance.optimizer["type"],
        "args" : argString
    }
    return outStr

def generateTrainingFiles(label, expInstances):
    os.mkdir(label)
    for instance in expInstances:

        filenameBase = label + "/" + label + "-" + str(instance.instanceIdx)

        with open(filenameBase + "-train" + ".py", "w+") as f:

            # Writes import statements to file
            f.write(imports(instance))
            f.write(optimizer(instance))