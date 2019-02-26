import json
import Experiment
import os
import docker_plugin
import slurm_plugin

## The ExperimentIntrepeter class is the main handler for the Experiment
# The interpreter takes in an experiment config JSON and generates an experiment
# Additionally, it interfaces with all of the artifact generation plugins to generate all of the experiment artifacts
# It then will execute the experiment by sending the jobs to the SLURM workload manager
# A single instance of the ExperimentInterpreter should be instantiated by the application server
class ExperimentInterpreter:
    ## __init__(self, artifactDir, mlFwk)
    # artifactDir - Directory in which to generate all of the experiment artifacts
    # mlFwk - The experimental framework being used for the experiments
    #           Currently, the Interpreter object stores the framework to avoid importing all of the framework plugins
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

    ## executeExperiment(self, expConfigFilepath)
    # This method should be called to run an experiment from a config JSON file
    # The method will instantiate the experiment object, generate all of the training artifacts, and send the jobs to SLURM
    # expConfigFilepath - Path to the config JSON
    def executeExperiment(self, expConfigFilepath):
        # Open and read in JSON file as dictionary
        with open(expConfigFilepath) as f:
            expConfig = json.load(f) # Load JSON as dict

        # Create an experiment object based on the expConfig file
        experiment = self.__generateExperimentObject__(expConfig)

        # Generate directories for experiment and experiment instances in the artifact directory
        self.__createExperimentDirs__(experiment)

        # Generate artifacts
        self.generateTrainingFiles(experiment)
        self.generateDockerFiles(experiment)
        self.generateSlurmBatchFile(experiment)


    ## __generateExperimentObject__(self, expConfig)
    # Generates an experiment object given a experiment config
    # expConfig - Dictionary containing contents of config JSON file
    def __generateExperimentObject__(self, expConfig):
        # Generate a dictionary containing slurm configuration from config file
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
