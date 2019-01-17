#Experiment class is the main class of the experimental framework
# An experiment object is created to define the various test cases to be executed
# This object will then produce the necessary configuration for jenkins and docker to execute all test cases and return the results
# Currently, an experiment only supports defining a static experiment. 
# This means that the entire experiemnt is decided before runtime and does not support features like hyperparameter search


from abc import ABC, abstractmethod


class ExperimentInstance():

    def __init__(self, mlFramework, instanceIdx, model, hyperparameters, dataset, modelLabel, hyperparametersLabel, datasetLabel):

        #Storing data as instance variables
        self.mlFramework = mlFramework
        self.dataset = dataset
        self.model = model
        self.hyperparameters = hyperparameters

        self.label = "Experiment Instance: %(instanceIdx)s\nModel: %(datasetLbl)s\nHyperparameterSet: %(modelLbl)s\nDataset: %(hpLbl)s" % {
            "instanceIdx" : instanceIdx,
            "modelLbl": str(modelLabel),
            "hpLbl" : str(hyperparametersLabel),
            "datasetLbl" : str(datasetLabel)
        }


    def getModel(self):
        return self.model
    
    def getDataset(self):
        return self.dataset

    def getHyperparameters(self):
        return self.hyperparameters

    def __str__(self):
        return self.label


class StaticExperiment():
    '''
    StaticExperiment is a base class to define an experiment involving various ML Models, hyperparameters, and datasets.
    The experiment is static because it does not support dynamically running experiment instances based on the results
    of previous instances. 
    '''

    def __init__(self, mlFramework, models = [], hyperparameterSets = [], datasets = []):
        #Defining parameters here
        self.framework = mlFramework
        self.models = models
        self.hyperparameterSets = hyperparameterSets
        self.datasets = datasets

    @abstractmethod
    def __getitem__(self, index):
        '''
        Returns an Experiment Instance object, which will be translated into a a file to execute training
        '''

        raise NotImplementedError

    @abstractmethod
    def __len__(self):

        raise NotImplementedError


    def getExperimentInstances(self):
        instances = []
        print(len(self))
        for idx in range(0,len(self)):
            instances.append(self[idx])
        return instances



class SimpleExperiment(StaticExperiment):


        def __init__(self, mlFramework, models = [], hyperparameterSets = [], datasets = []):
            super().__init__(mlFramework, datasets, models, hyperparameterSets)

        
        def __len__(self):
            return len(self.datasets)*len(self.models)*len(self.hyperparameterSets)
        
        def __getitem__(self, index):
            if(index >= len(self)):
                #Todo: Define this exception
                raise Exception()
            else:
                tempIdx = index
                datasetIdx = tempIdx%len(self.datasets)
                tempIdx = int(tempIdx/len(self.datasets))

                hyperparameterSetIdx = tempIdx%len(self.hyperparameterSets)
                tempIdx = int(tempIdx/len(self.hyperparameterSets))

                modelIdx = tempIdx%len(self.models)
                
                return ExperimentInstance("keras",
                    index,
                    self.datasets[datasetIdx],
                    self.models[modelIdx],
                    self.hyperparameterSets[hyperparameterSetIdx],
                    datasetIdx,
                    modelIdx,
                    hyperparameterSetIdx
                )

            

exp = SimpleExperiment('kera', [0,1],[0,1],[0,1])

instances = exp.getExperimentInstances()

for instance in instances:
    print(instance)
    print('\n')




