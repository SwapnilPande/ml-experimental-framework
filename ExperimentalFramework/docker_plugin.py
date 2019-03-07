import os
import Experiment


class DockerPlugin:

    def __init__(self):
        # Define the mount directores for the data and experiment
        self.datasetMountDir = "/data"
        self.artifactMountDir = "/experiment"


    def fromContainer(self, frameworkContainer):
        return "FROM " + frameworkContainer + "\n\n"

    def installDependencies(self, dependencies = []):
        outStr = ""
        if( dependencies != []):
            outStr = "CMD apt update\n"
            outStr += "CMD apt install -y "
            outStr += " ".join(dependencies)
            outStr += "\n\n"
        return outStr

    def runTrainFile(self, instance):
        _,file = os.path.split(instance.trainfile)
        outStr = "CMD python {trainfile}".format(trainfile=file)
        return outStr

    def shebang(self):
        return "#!/bin/bash\n\n"

    def dockerBuild(self, label, instance):
        instanceDir,instanceName = os.path.split(instance.trainfile)
        return "docker build --tag={label}-{instanceIdx} {instanceDir}\n\n".format(
            label = label,
            instanceIdx = instance.instanceIdx,
            instanceDir = instanceDir)

    def dockerRun(self, label, expArtifactDir, instance):
        runArgs = "--rm -v {dataDir}:{dataMountDir} -v {experimentDir}:{experimentMountDir}".format(
            dataDir = instance.getDataset()["path"],
            dataMountDir = self.datasetMountDir,
            experimentDir = expArtifactDir,
            experimentMountDir = self.artifactMountDir
        )
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
                f.write(self.runTrainFile(instance))
            instance.dockerfile = dockerfileName

            executedockerfileName = (os.path.join(instance.artifactDir, "run_dockerfile")).format(
                instanceIdx = instance.instanceIdx
            )

            with open(executedockerfileName, "w+") as f:
                f.write(self.shebang())
                f.write(self.dockerBuild(experiment.label, instance))
                f.write(self.dockerRun(experiment.label, experiment.artifactDir, instance))
            instance.executeFile = executedockerfileName


        return experiment