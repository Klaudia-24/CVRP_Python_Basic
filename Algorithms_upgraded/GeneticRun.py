import itertools
from Data.DataParser import parseData
import glob
from Algorithms_upgraded.Genetic import Genetic, Mutation
import timeit
from datetime import datetime

DIR_PATH = "..\Data\RawData\**\*.*"

GENETIC_PARAMETERS = list(
    # itertools.product(*[[0.05, 0.25], [0.01, 0.2], [5, 10, 20, 40, 80, 160], [10, 20, 40, 80, 160, 320]]))
    # itertools.product(*[[0.05, 0.25], [0.01, 0.2], [5, 10, 20, 40, 80, 160, 320, 640], [10, 20, 40, 80, 160, 320, 640]]))
    itertools.product(*[[0.25], [0.2], [160], [320]]))


TEST_ITERATIONS = 5


def main():
    for file in glob.glob(DIR_PATH, recursive=True):
        resultName = file.split("\\")[-1].split(".")[0]
        metaData, nodes, demand = parseData(file)
        algGen = Genetic(metaData, nodes, demand, 0, 0, 0, 0, testIterCount=TEST_ITERATIONS)

        algGen.setNCross(2)
        algGen.setMutation(Mutation.BESTSOLUTION)

        resultFileName = f"../cmtResultGen/result{resultName}_{algGen.nCross}_{algGen.mutationMode.value}_2.xml"

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


if __name__ == "__main__":
    main()
