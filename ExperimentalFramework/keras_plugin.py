import os
import Experiment

class plugin:
    def formatArgs(self, argDict):
        argList = []
        # Convert dictionary of arguments into string of comma seperated arguments
        for key,value in argDict.items():
            argStr = str(key) + " = "
            if(type(value) is str):
                # Add quotes to strings
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

        # Import train data generator
        outStr += "import " + expInstance.dataset["train"]["generator"] + " as train_generator\n"

        # Import validation data generator
        if("validation" in expInstance.dataset):
            outStr += "import " + expInstance.dataset["validation"]["generator"] + " as validation_generator\n"

        # Import test data generator
        if("test" in expInstance.dataset):
            outStr += "import " + expInstance.dataset["test"]["generator"] + " as test_generator\n"

        outStr += "\n"

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
        # Initialize preprocessor objects
        outStr = "# Initialize pre-processor\n"
        outStr += "trainDataGen = train_generator.getGenerator({args})\n".format(
                preprocessor = expInstance.dataset["train"]["generator"],
                args = self.formatArgs(expInstance.dataset["train"]["args"])
            )
        if("validation" in expInstance.dataset):
            outStr += "validationDataGen = validation_generator.getGenerator({args})\n".format(
                preprocessor = expInstance.dataset["validation"]["generator"],
                args = self.formatArgs(expInstance.dataset["validation"]["args"])
            )
        if("test" in expInstance.dataset):
            outStr += "testDataGen = test_generator.getGenerator({args})\n".format(
                preprocessor = expInstance.dataset["test"]["generator"],
                args = self.formatArgs(expInstance.dataset["test"]["args"])
            )
        outStr += "\n"

        return outStr

    def model(self, expInstance):
        # Compile Model
        outStr = "# Compile Model\n"
        outStr += "model = %(model)s.getModel()\n" % { "model" : expInstance.model["path"]}
        outStr += "model.compile(optimizer, %(args)s)\n\n" % {"args" : self.formatArgs(expInstance.model["model_parameters"])}
        return outStr

    def fitGenerator(self, expInstance):
        if("validation" in expInstance.dataset):
            outStr = "model.fit_generator(trainDataGen, validation_data = validationDataGen,  {args})\n\n".format(args = self.formatArgs(expInstance.hyperparameters))
        else:
            outStr = "model.fit_generator(trainDataGen,  {args})\n\n".format(args = self.formatArgs(expInstance.hyperparameters))
        return outStr

    def evaluateGenerator(self, expInstance):
        outStr = ""
        if("test" in expInstance.dataset):
            outStr += "# Test Model\n"
            outStr += "testResult = model.evaluate_generator(testDataGen)\n"
            outStr += "print(\"Final Test Loss: \" + str(testResult))\n\n"
        return outStr

    def getRequirements(self):
        return ["keras"]

    def generateTrainingFiles(self, experiment):
        for instance in experiment.expInstances:
            # Construct full filename and path for python training file
            filename = (os.path.join(instance.artifactDir,"train-{label}-{instanceIdx}.py")).format(
                label = experiment.label,
                instanceIdx = instance.instanceIdx
            )

            # Open python training file
            with open(filename, "w+") as f:
                # Writes import statements to file
                f.write(self.imports(instance))
                f.write(self.optimizer(instance))
                f.write(self.preprocessor(instance))
                f.write(self.model(instance))
                f.write(self.fitGenerator(instance))
                f.write(self.evaluateGenerator(instance))

            instance.trainfile = filename
        return experiment
