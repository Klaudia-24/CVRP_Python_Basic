import itertools
from Data.DataParser import parseRawData
import glob
from Algorithms_upgraded.Genetic import Genetic, Mutation
import timeit
from datetime import datetime
from Data.DataAnalyser import parseData, ALGTYPE

# DIR_PATH = "..\Data\RawData\**\*.*"

dir_path_local = "..\Data\RawData\**\*.*"
dir_path_folder = "..\\Data\\RawData\\**\\"
SOL_PATH = "..\Data\ParsedData\**\*.*"


GENETIC_PARAMETERS = list(
    itertools.product(*[[0.05, 0.25], [0.01, 0.2], [5, 10, 20, 40, 80, 160], [10, 20, 40, 80, 160, 320]]))
    # itertools.product(*[[0.05, 0.25], [0.01, 0.2], [5, 10, 20, 40, 80, 160, 320, 640], [10, 20, 40, 80, 160, 320, 640]]))
    # itertools.product(*[[0.05], [0.2], [80], [320]]))

# TEST_ITERATIONS = 5

TEST_ITERATIONS = 20

def main():
    for file in glob.glob(dir_path_local, recursive=True):
        resultName = file.split("\\")[-1].split(".")[0]
        metaData, nodes, demand = parseRawData(file)
        algGen = Genetic(metaData, nodes, demand, 0, 0, 0, 0, testIterCount=TEST_ITERATIONS)

        algGen.setNCross(1)
        algGen.setMutation(Mutation.RANDOM)

        # algGen.setNCross(2)
        # algGen.setMutation(Mutation.RANDOM)

        # algGen.setNCross(1)
        # algGen.setMutation(Mutation.BESTSOLUTION)

        # algGen.setNCross(2)
        # algGen.setMutation(Mutation.BESTSOLUTION)

        # resultFileName = f"../cmtResultGen/iter_20/result{resultName}_{algGen.nCross}_{algGen.mutationMode.value}.xml"

        resultFileName = f"../Data/ParsedData/Genetic/result{resultName}_{algGen.nCross}_{algGen.mutationMode.value}.xml"

        # resultsFile = open(f"Data/ParsedData/{resultName}_{algGen.nCross}_{algGen.mutationMode.value}.xml", "a+")
        resultsFile = open(resultFileName, "a+")
        resultsFile.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        resultsFile.write('<results>\n')
        resultsFile.close()
        print("")
        print(f"{resultName}.xml")

        start_time = datetime.now().strftime("%H:%M:%S")

        index = 0
        for args in GENETIC_PARAMETERS:
            index += 1
            algGen.setParameters(*args)
            print(f"{index}/{len(GENETIC_PARAMETERS)}     {resultName}.vrp")

            now = datetime.now()
            test_time = now.strftime("%H:%M:%S")
            print(f"Start time: {start_time}  |  {test_time}")

            time = timeit.timeit(algGen.run, number=1) / TEST_ITERATIONS
            res = algGen.returnBest()
            resultsFile = open(resultFileName, "a+")
            resultsFile.write(
                f'<test crossOption="{algGen.crossRatio}" mutationOption="{algGen.mutationRatio}" population="{algGen.popCount}" '
                f'generationCount="{algGen.genCount}" avgTime="{time}">\n')
            for i in range(TEST_ITERATIONS):
                resultsFile.write(f'<route id="{i}" score="{res[i].score}">\n')
                resultsFile.write(str(res[i].route) + "\n")
                resultsFile.write('</route>\n')
            resultsFile.write('</test>\n')
            resultsFile.close()
        resultsFile = open(resultFileName, "a+")
        resultsFile.write('</results>\n')
        resultsFile.close()


def analyseData(mode: ALGTYPE):
    resultsFile = open(f"../Results/bestPathData.txt", "w")
    index = 0
    count = len(glob.glob(SOL_PATH, recursive=True))
    for file in glob.glob(SOL_PATH, recursive=True):
        index += 1
        print(f"{index}/{count} Parsing: " + file)
        result = parseData(file, DIR_PATH=dir_path_folder, mode=mode)
        resultName = file.split("\\")[-1].split(".")[0]
        resultsFile.write(file + "\n")
        resultsFile.write(result.bestRoute().__repr__() + "\n")
        folder = "Genetic" if ALGTYPE.GENETIC else "Ant"
        i = 0
        for _ in glob.glob(f"..\\Results\\{folder}\\{resultName}_graph.jpg"):
            i += 1
        if i != 0:
            resultName += f"_({i})"
        result.bestRouteGraph(f"..\\Results\\{folder}\\{resultName}_graph.jpg")
        # result.timeGraphFor(path=f"..\\Results\\{folder}\\{resultName}_time.jpg")
        result.timeGraph(path=f"..\\Results\\{folder}\\{resultName}_timeAllGen.jpg")
        result.scoreGraph(path=f"..\\Results\\{folder}\\{resultName}_score.jpg")
        result.lostCapacityGraph(path=f"..\\Results\\{folder}\\{resultName}_capacityLost.jpg")


if __name__ == "__main__":
    main()

    # analyseData(ALGTYPE.GENETIC)
