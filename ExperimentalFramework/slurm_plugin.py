import os

class SlurmPlugin:
    def formatArgs(self, argDict):
        argList = []
        # Convert dictionary of arguments into string of comma seperated arguments
        for key,value in argDict.items():
            argStr = str(key) + "="
            if(type(value) is str):
                argStr += "\"" + value + "\""
            else:
                argStr += str(value)
            argList.append(argStr)
        return " ".join(argList)

    def shebang(self):
        return "#!/bin/bash\n\n"

    def srun(self, instance, slurmConfig):
        outStr = "srun "
        if(slurmConfig != {}):
            outStr += self.formatArgs(slurmConfig)
        outStr += " "
        outStr += instance.executeFile
        outStr += " &\n\n"
        return outStr

    def generateBatchFile(self, experiment, expInstances):
        filename = experiment.artifactDir + "/run_experiment"
        with open(filename, "w+") as f:
            f.write(self.shebang())

            for instance in expInstances:
                f.write(self.srun(instance, experiment.slurmConfig))

            return expInstances