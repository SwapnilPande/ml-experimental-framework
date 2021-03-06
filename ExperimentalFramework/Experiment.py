## @package Experiment
# The Experiment module contains multiple classes for defining experiments and experiment instances
# An Experiment object contains 'independent variables' that will be tested and generates ExperimentInstance object,
# which are specific combinations of the independent variables to execute.



from abc import ABC, abstractmethod

## ExperimentInstance
# An ExperimentInstance objects contains a specific combination of the 'independent variables' of the experiment, as well as a label describing the instance
# The ExperimentInstance object will be converted into a training program by the specific plugin for the ML Framework
class ExperimentInstance():
    """
        instanceIdx - Unique index assigned to the experiment instance. Experiment label + instanceIdx provide a unique identifier for instance
        mlFramework - Name of machine learning framework to use, name must match plugin name exactly
        slurmConfig - Dictionary containing the arguments to pass to SLURM when executing this instance as defined in the config JSON folder.
                        This should not include the "--output" parameter (which will be added in the init)
        artifactDir - Directory in which all the training artifacts are generated for the experiment instance
        instanceIdx - Unique index assigned to instance. The experiment label + instance index define a unique identifier for each experiment instance
        model - python module path containing the model associated with instance
        hyperparameters - dictionary containing hyperparameters associated with model
        dataset - path to dataset to train model
        optimizer - python module path containing the optimizer associated with the instance
        modelLabel - Human readable label assigned to the model
        hyperparameterLabel - Human readable label assigned to the hyperparameter set
        datasetLabel - Human readable label assigned to dataset
        optimizerLabel - Human readable label assigned to optimizer
    """
    def __init__(self, instanceIdx, mlFramework, artifactDir, slurmConfig, model, hyperparameters, dataset, optimizer, modelLabel, hyperparametersLabel, datasetLabel, optimizerLabel):
        #Storing all arguments as instance variables for the ExperimentInstance
        self.instanceIdx = instanceIdx
        self.mlFramework = mlFramework
        self.artifactDir = artifactDir.format(instanceIdx = instanceIdx)
        self.dataset = dataset
        self.model = model
        self.hyperparameters = hyperparameters
        self.optimizer = optimizer
        self.slurmConfig = slurmConfig

        # Variables for all files associated with experiment
        self.trainfile = None # Python file for training the model. This field is populated by the ML Framework plugin
        self.dockerfile = None # Dockerfile associated with instance. This field is populated by the dockerfile generator
        self.executeFile = None # Bash script to build and run docker container. THis field is populated by the dockerfile generator
        self.outputFile = "instance_{instanceIdx}.out".format(instanceIdx=self.instanceIdx) # File to redirect stdout/stderr, defined relative to artifactDir

        # Generate combined label to describe experiment instance
        self.label = "Experiment Instance: {instanceIdx}\nModel: {modelLbl}\nOptimizer: {optimizerLbl}\nHyperparameterSet: {hpLbl}\nDataset: {datasetLbl}\n".format(
            instanceIdx = instanceIdx,
            modelLbl = str(modelLabel),
            hpLbl = str(hyperparametersLabel),
            datasetLbl = str(datasetLabel),
            optimizerLbl = str(optimizerLabel))


    def getModel(self):
        return self.model

    def getDataset(self):
        return self.dataset

    def getHyperparameters(self):
        return self.hyperparameters

    def __str__(self):
        return self.label

## StaticExperiment
# StaticExperiment is a base class to define an experiment involving various ML Models, hyperparameters, and datasets.
# THe ExperimentInstance objects are all generated by creating combinations of these variables. The ExperimentInstance
# objects are all created at the beginning of the experiment execution, and therefore, cannot be created or changed
# based on the results from the experimentation, hence the name static. A DynamicExperiment can be used for applications
# such as hyperparameter search
class StaticExperiment():
    """
        label - Name assigned to the experiment in the config file
        mlFramework - Name of machine learning framework to use, name must match plugin name exactly
        requirements - Directory to requirements.txt
        slurmConfig - Dictionary containing the arguments to pass to SLURM when executing this instance as defined in the config JSON folder.
                        This should not include the "--output" parameter (which will be added in the experimentInstance)
        artifactDir - Directory path in which all the training artifacts are generated for the experiment
        experimentDir - Directory containing the ExperimentConfig.json file and all associated files

        models - Dictionary of the models from the config json file
        hyperparameterSets - Dictionary of the hyperparameterSets from config json file
        datasets - Dictionary of the datsets from the config json file
        optimizers - Dictionary of the optimizers from the config json file
    """
    def __init__(self, label, mlFramework, requirements, slurmConfig, artifactDir, experimentDir, models, hyperparameterSets, datasets, optimizers):
        # Save arguments as instance variables
        self.label = label
        self.mlFramework = mlFramework

        self.requirements = requirements
        self.artifactDir = artifactDir
        self.experimentDir = experimentDir

        """
            Path describing artifact directory for each experiment instance. This is a generic path describing the directory pattern
            for all instances. This path contains {instanceIdx}. When the experiment instance is created, the instanceIdx is
            populated into the path and stored in the instance artifactDir.
        """
        self.expInstanceDir = artifactDir + "/instance_{instanceIdx}/"
        self.executeFile = None # Path to file to run entire experiment (populated by SLURM plugin)
        self.frameworkContainer = None # Name of the docker container to pull associated with the ML experimental framework. This field is populated by the ML Framework plugin


        self.slurmConfig = slurmConfig
        self.models = models
        self.hyperparameterSets = hyperparameterSets
        self.datasets = datasets
        self.optimizers = optimizers

        self.expInstances = self.__generateInstances__()



    @abstractmethod
    def __generateInstances__(self):
        """
            Generates the experiment instances for the static experiment
        """
        raise NotImplementedError

    def __getitem__(self, index):
        if(index >= len(self)):
                #Todo: Define this exception
                raise Exception()
        else:
            return(self.expInstances[index])

    @abstractmethod
    def __len__(self):
        """
            Returns the number of experiment instances that will be generated
        """
        raise NotImplementedError


    # Returns a list containing all of the experiment instance objects
    def getExperimentInstances(self):
        return self.expInstances


## SimpleStaticExperiment
# The SimpleStaticExperiment is a simple example of an implementation of the StaticExperiment.
# The various models, hyperparameters, and datasets are passed when the SimpleStaticExperiment is instantiated.
# The __getitem__ method will simply return every combination of the 4 independent variables to be executed.
class SimpleStaticExperiment(StaticExperiment):

        # Initialize SimpleStaticExperiment
        # No additional arguments to the base class (all arguments described in StaticExperiment)
        def __init__(self, label, mlFramework, requirements, slurmConfig, artifactDir, experimentDir, models, hyperparameterSets, datasets, optimizers):
            super().__init__(label, mlFramework, requirements, slurmConfig, artifactDir, experimentDir, models,  hyperparameterSets, datasets, optimizers)

        def __generateInstances__(self):
            instances = []
            for i in range(self.__len__()):
                tempIdx = i

                datasetIdx = tempIdx%len(self.datasets)
                tempIdx = int(tempIdx/len(self.datasets))

                hyperparameterSetIdx = tempIdx%len(self.hyperparameterSets)
                tempIdx = int(tempIdx/len(self.hyperparameterSets))

                optimizerIdx = tempIdx%len(self.optimizers)
                tempIdx = int(tempIdx/len(self.optimizers))

                modelIdx = tempIdx%len(self.models)

                instances.append(ExperimentInstance(i,
                    self.mlFramework,
                    self.expInstanceDir,
                    self.slurmConfig,
                    self.models[modelIdx],
                    self.hyperparameterSets[hyperparameterSetIdx]["hyperparameters"],
                    self.datasets[datasetIdx],
                    self.optimizers[optimizerIdx]["optimizer"],
                    self.models[modelIdx]["label"],
                    self.hyperparameterSets[hyperparameterSetIdx]["label"],
                    self.datasets[datasetIdx]["label"],
                     self.optimizers[optimizerIdx]["label"])
                )
            return instances


        def __len__(self):
            return len(self.datasets)*len(self.models)*len(self.hyperparameterSets)*len(self.optimizers)




