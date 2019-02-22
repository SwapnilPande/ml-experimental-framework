import json
import Experiment
import os
import docker_plugin
import slurm_plugin

class ExperimentInterpreter:

    def __init__(self, artifactDir, mlFwk):
        # Save the directory in which all of the artifacts for the experiments will be generated
        self.artifactDir = artifactDir

        self.mlFwk = mlFwk
        # Imports the necessary ML framework plugin as defined in config
        self.pluginName = mlFwk + "_plugin"
        self.mlFwkPluginImport = __import__(self.pluginName)
        print("Debug - Machine Learning Framework : " + self.mlFwk)

        #Initialize the plugin object from the ML Framework file
        self.mlFwkPlugin = self.mlFwkPluginImport.plugin()
        self.dockerPlugin = docker_plugin.DockerPlugin()
        self.slurmPlugin = slurm_plugin.SlurmPlugin()


    def executeExperiment(self, expConfigFilepath):
        # Open and read in JSON file as dictionary
        with open(expConfigFilepath) as f:
            expConfig = json.load(f) # Load JSON as dict

        # Create an experiment object based on the expConfig file
        experiment = self.__generateExperimentObject__(expConfig)

        # Generate directories for experiment and experiment instances in the artifact directory
        self.__createExperimentDirs__(experiment)

        self.generateTrainingFiles(experiment)
        self.generateDockerFiles(experiment)
        self.generateSlurmBatchFile(experiment)


    def __generateExperimentObject__(self, expConfig):

        # Generate a dictionary containing slurm configuration from config file
        # Does not include the output parameter

        expDir = self.__generateUniqueExperimentDir__(expConfig["label"])

        #Check if experiment type is static simple. If it is, create an experiment object and generate the instances
        if(expConfig["experiment_type"] == "static_simple"):
            experiment = Experiment.SimpleStaticExperiment(expConfig["label"],
            self.mlFwk,
            expConfig["resources"],
            expDir,
            expConfig["models"],
            expConfig["hyperparameter_sets"],
            expConfig["datasets"],
            expConfig["optimizers"])

        # Return the generated experiment object
        return experiment

    def __generateUniqueExperimentDir__(self, expLabel):
        expDir = os.path.join(self.artifactDir,expLabel)
        # TODO - generate unique hash for each file
        return expDir


    def __createExperimentDirs__(self, experiment):
        os.mkdir(experiment.artifactDir)
        for instance in experiment.expInstances:
            os.mkdir(instance.artifactDir)


    def generateTrainingFiles(self, experiment):
        return self.mlFwkPlugin.generateTrainingFiles(experiment)

    def generateDockerFiles(self, experiment):
        self.instances = self.dockerPlugin.generateDockerFiles(experiment, "kerasContainer")

    def generateSlurmBatchFile(self, experiment):
        self.experiment = self.slurmPlugin.generateBatchFile(experiment)


if(__name__ == "__main__"):
    expInterpret = ExperimentInterpreter("test", "keras")
    expInterpret.executeExperiment("sampleconfig.json")
