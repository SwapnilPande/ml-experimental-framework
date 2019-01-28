import json
import Experiment

class ExperimentInterpreter:

    def __init__(self, expConfigFilepath):
        # Open and read in JSON file as dictionary
        self.expConfigFilepath = expConfigFilepath
        with open(expConfigFilepath) as f:
            self.expConfig = json.load(f) # Load JSON as dict

        self.expLabel = self.expConfig["label"]

        #Save framework models, hyperparameter sets, and datasets
        self.mlFwk = self.expConfig["ml_framework"]
        self.models = self.expConfig["models"]
        self.hyperparameterSets = self.expConfig["hyperparameter_sets"]
        self.datasets = self.expConfig["datasets"]
            
        # Imports the necessary ML framework plugin as defined in config
        self.pluginName = self.mlFwk + "_plugin"
        self.mlFwkPlugin = __import__(self.pluginName)

        print("Debug - Machine Learning Framework : " + self.expConfig["ml_framework"])

        #Check if experiment type is static simpe. If it is, create an experiment object and generate the instances
        if(self.expConfig["experiment_type"] == "static_simple"):
            self.experiment = Experiment.SimpleStaticExperiment(self.mlFwk, self.models, self.hyperparameterSets, self.datasets)
            self.instances = self.experiment.getExperimentInstances()
        
    def generateTrainingFiles(self):
        self.mlFwkPlugin.generateTrainingFiles(self.expLabel, self.instances)
        


        

        
if(__name__ == "__main__"):
    expInterpret = ExperimentInterpreter("sampleconfig.json")
    expInterpret.generateTrainingFiles()
