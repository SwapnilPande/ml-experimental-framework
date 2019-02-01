import os
import Experiment


def formatArgs(argDict):
    argList = []
    # Convert dictionary of arguments into string of comma seperated arguments
    for key,value in argDict.items():
        argStr = str(key) + " = "
        if(type(value) is str):
            argStr += "\"" + value + "\""
        else:
            argStr += str(value)
        argList.append(argStr)
    return ", ".join(argList)

# Returns the strings to write to import necessary packages
def imports(expInstance):
    outStr = "# Imports"
    outStr += "from keras.models import Model\n"
    #Import the correct optimizer for this model
    outStr += "from keras.optimizers import %(type)s\n" % {"type" : expInstance.optimizer["type"]}
    outStr += "import " + expInstance.model["path"] + "\n\n"
    return outStr

def optimizer(expInstance):
    # Generate the optimizer object with the correct parameters
    outStr = "# Define optimizer\n"
    outStr += "optimizer = %(type)s(%(args)s)\n\n" % {
        "type" : expInstance.optimizer["type"],
        "args" : formatArgs(expInstance.optimizer["optimizer_parameters"])
    }
    return outStr

def model(expInstance):
    # Compile Model
    outStr = "#Compile Model\n"
    outStr += "model = %(model)s.getModel()\n" % { "model" : expInstance.model["path"]}
    outStr += "model.compile(optimizer, %(args)s)\n\n" % {"args" : formatArgs(expInstance.model["model_parameters"])}
    return outStr

def fitGenerator(expInstance):
    outStr = ""
    return outStr

def generateTrainingFiles(label, expInstances):
    os.mkdir(label)
    for instance in expInstances:

        filenameBase = label + "/" + label + "-" + str(instance.instanceIdx)

        with open(filenameBase + "-train" + ".py", "w+") as f:

            # Writes import statements to file
            f.write(imports(instance))
            f.write(optimizer(instance))
            f.write(model(instance))
