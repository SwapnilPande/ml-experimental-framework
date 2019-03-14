import os
import Experiment


class DockerPlugin:

    def __init__(self):
        # Define the mount directores for the data and experiment
        self.datasetMountDir = "/data"
        self.artifactMountDir = "/artifacts"
        self.experimentMountDir = "/experiment"


    def fromContainer(self, frameworkContainer):
        return "FROM " + frameworkContainer + "\n\n"

    def installDependencies(self, dependencies = []):
        outStr = ""
        if( dependencies != []):
            outStr = "RUN apt update\n"
            outStr += "RUN apt upgrade -y\n"
            outStr += "RUN apt install -y "
            outStr += " ".join(dependencies) + '\n'
            outStr += "\n\n"
        return outStr

    def installPipRequirements(self, requirements):
        return "RUN pip install {requirements}\n\n".format(requirements = " ".join(requirements))

    def pythonEnvironment(self):
        return "ENV PYTHONPATH \"${{PYTHONPATH}}:{experimentMountDir}\"\n\n".format(experimentMountDir = self.experimentMountDir)

    def runTrainFile(self, instance):
        _,file = os.path.split(instance.trainfile)
        return "CMD python {trainfile}".format(trainfile=os.path.join(self.artifactMountDir, file))

    def shebang(self):
        return "#!/bin/bash\n\n"

    def dockerBuild(self, label, instance):
        instanceDir,instanceName = os.path.split(instance.trainfile)
        return "docker build --tag={label}-{instanceIdx} {instanceDir}\n\n".format(
            label = label,
            instanceIdx = instance.instanceIdx,
            instanceDir = instanceDir)

    #TODO: CHANGE RUNTIME FROM MAGIC VALUE
    def dockerRun(self, label, expDir, instance, runtime="nvidia"):
        runArgs = "--rm -v {artifactDir}:{artifactMountDir} -v {experimentDir}:{experimentMountDir}".format(
            artifactDir = instance.artifactDir,
            artifactMountDir = self.artifactMountDir,
            experimentDir = expDir,
            experimentMountDir = self.experimentMountDir
        )
        if(instance.getDataset()["path"] != ""):
            runArgs += " -v {datasetDir}:{datasetMountDir}".format(
                datasetDir = instance.getDataset()["path"],
                datasetMountDir = self.datasetMountDir
            )
        if(runtime != None):
            runArgs += " --runtime={runtime}".format(runtime = runtime)


        return "docker run {args} {label}-{instanceIdx} ".format(
            label = label,
            instanceIdx = instance.instanceIdx,
            args = runArgs)

    def generateDockerFiles(self, experiment, frameworkContainer):
        for instance in experiment.expInstances:
            # Construct full filename and path for python training file
            dockerfileName = (os.path.join(instance.artifactDir, "Dockerfile")).format(
                instanceIdx = instance.instanceIdx
            )

             # Open dockerfile
            with open(dockerfileName, "w+") as f:
                f.write(self.fromContainer(frameworkContainer))
                f.write(self.installDependencies(["libsm6", "libxext6"]))
                f.write(self.installPipRequirements(experiment.requirements))
                f.write(self.pythonEnvironment())
                f.write(self.runTrainFile(instance))
            instance.dockerfile = dockerfileName

            executedockerfileName = (os.path.join(instance.artifactDir, "run_dockerfile")).format(
                instanceIdx = instance.instanceIdx
            )

            with open(executedockerfileName, "w+") as f:
                f.write(self.shebang())
                f.write(self.dockerBuild(experiment.label, instance))
                f.write(self.dockerRun(experiment.label, experiment.experimentDir, instance))
            instance.executeFile = executedockerfileName


        return experiment