import json
import Experiment
import os
import docker_plugin

class ExperimentInterpreter:

    def __init__(self, expConfigFilepath, artifactDir):
        # Open and read in JSON file as dictionary
        self.expConfigFilepath = expConfigFilepath
        with open(expConfigFilepath) as f:
            self.expConfig = json.load(f) # Load JSON as dict

        self.expLabel = self.expConfig["label"]

        #Save framework models, hyperparameter sets, datasets, and optimizers
        self.mlFwk = self.expConfig["ml_framework"]
        self.models = self.expConfig["models"]
        self.hyperparameterSets = self.expConfig["hyperparameter_sets"]
        self.datasets = self.expConfig["datasets"]
        self.optimizers = self.expConfig["optimizers"]

        # Imports the necessary ML framework plugin as defined in config
        self.pluginName = self.mlFwk + "_plugin"
        self.mlFwkPluginImport = __import__(self.pluginName)

        #Initialize the plugin object from the ML Framework file
        self.mlFwkPlugin = self.mlFwkPluginImport.plugin()
        self.dockerPlugin = docker_plugin.DockerPlugin()

        print("Debug - Machine Learning Framework : " + self.expConfig["ml_framework"])

        #Check if experiment type is static simpe. If it is, create an experiment object and generate the instances
        if(self.expConfig["experiment_type"] == "static_simple"):
            self.experiment = Experiment.SimpleStaticExperiment(self.mlFwk, self.models, self.hyperparameterSets, self.datasets, self.optimizers)
            self.instances = self.experiment.getExperimentInstances()
            # for instance in self.instances:
            #     #print(instance)

        self.artifactDir = (artifactDir + "/{label}").format(label = self.expLabel)
        self.experimentDir = self.artifactDir + "/instance_{instanceIdx}/"
       # print(self.experimentDir)

    def initializeExpDirs(self):
        os.mkdir(self.artifactDir)
        for instance in self.instances:
            os.mkdir(self.experimentDir.format(instanceIdx = instance.instanceIdx))


    def generateTrainingFiles(self):
        self.instances = self.mlFwkPlugin.generateTrainingFiles(self.expLabel, self.instances, self.experimentDir)

    def generateDockerFiles(self):
        self.instances = self.dockerPlugin.generateDockerFiles("kerasContainer", self.instances, self.experimentDir)


if(__name__ == "__main__"):
    expInterpret = ExperimentInterpreter("sampleconfig.json","test")
    expInterpret.initializeExpDirs()
    expInterpret.generateTrainingFiles()
    expInterpret.generateDockerFiles()
