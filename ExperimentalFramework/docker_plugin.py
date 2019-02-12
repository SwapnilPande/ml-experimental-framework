import os

class DockerPlugin:
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


    def generateDockerFiles(self, frameworkContainer, expInstances, instanceDir):
        for instance in expInstances:
            # Construct full filename and path for python training file
            filename = (instanceDir + "Dockerfile").format(
                instanceIdx = instance.instanceIdx
            )

             # Open dockerfile
            with open(filename, "w+") as f:
                f.write(self.fromContainer(frameworkContainer))
                f.write(self.installDependencies(["libsm6", "libxext6"]))
                f.write(self.runTrainFile(instance))
            instance.dockerfile = filename
        return expInstances