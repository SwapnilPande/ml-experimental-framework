import os
import Experiment

class plugin:
    def formatArgs(self, argDict):
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
    def imports(self, expInstance):
        outStr = "# Imports"
        outStr += "from keras.models import Model\n"
        #Import the correct optimizer for this model
        outStr += "from keras.optimizers import %(type)s\n" % {"type" : expInstance.optimizer["type"]}
        outStr += "import " + expInstance.model["path"] + "\n"
        outStr += "import " + expInstance.dataset["preprocessor"] + "\n\n"
        return outStr

    def optimizer(self, expInstance):
        # Generate the optimizer object with the correct parameters
        outStr = "# Define optimizer\n"
        outStr += "optimizer = %(type)s(%(args)s)\n\n" % {
            "type" : expInstance.optimizer["type"],
            "args" : self.formatArgs(expInstance.optimizer["optimizer_parameters"])
        }
        return outStr

    def preprocessor(self, expInstance):
        outStr = "# Initialize pre-processor\n"
        outStr += "preprocessor = %(preprocessor)s.getPreprocessor()\n\n" % { "preprocessor" : expInstance.dataset["preprocessor"]}
        return outStr

    def model(self, expInstance):
        # Compile Model
        outStr = "# Compile Model\n"
        outStr += "model = %(model)s.getModel()\n" % { "model" : expInstance.model["path"]}
        outStr += "model.compile(optimizer, %(args)s)\n\n" % {"args" : self.formatArgs(expInstance.model["model_parameters"])}
        return outStr

    def fitGenerator(self, expInstance):
        outStr = "model.fit_generator(preprocessor, %(args)s)\n\n" % {"args" : self.formatArgs(expInstance.hyperparameters) }

        return outStr

    def getRequirements(self):
        return ["keras"]

    def generateTrainingFiles(self, label, expInstances, instanceDir):
        for instance in expInstances:
            # Construct full filename and path for python training file
            filename = (instanceDir + "train-{label}s-{instanceIdx}.py").format(
                label = label,
                instanceIdx = instance.instanceIdx
            )

            with open(filename, "w+") as f:

                # Writes import statements to file
                f.write(self.imports(instance))
                f.write(self.optimizer(instance))
                f.write(self.preprocessor(instance))
                f.write(self.model(instance))
                f.write(self.fitGenerator(instance))
