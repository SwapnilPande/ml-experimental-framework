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
        print(slurmConfig)
        outStr = "srun "
        if(slurmConfig != {}):
            outStr += self.formatArgs(slurmConfig)
        outStr += " "
        outStr += instance.executeFile
        outStr += " &\n\n"
        return outStr

    def generateBatchFile(self, experiment):
        filename = experiment.artifactDir + "/run_experiment"
        with open(filename, "w+") as f:
            f.write(self.shebang())

            for instance in experiment.expInstances:
                slurmArgs = {
                    "--nodes" : 1,
                    "--ntasks" : 1,
                    "--cpus-per-task" : instance.slurmConfig["cpus"],
                    "--mem" : instance.slurmConfig["memory"],
                    "--gres" : "gpu:" + str(instance.slurmConfig["gpus"]),
                    "--output" :  os.path.join(instance.artifactDir,instance.outputFile)
                }
                f.write(self.srun(instance, slurmArgs))
        experiment.executeFile = filename
        return experiment