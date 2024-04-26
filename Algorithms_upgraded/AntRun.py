import itertools
from Data.DataParser import parseRawData
import glob
import timeit
from Data.DataAnalyser import parseData, ALGTYPE
from Algorithms_upgraded.Ant import Ant
from datetime import datetime

dir_path_local = "..\Data\RawData\**\*.*"
dir_path_folder = "..\\Data\\RawData\\**\\"
SOL_PATH = "..\Data\ParsedData\**\*.*"

#                                            antCount            iterCount
# ANT_PARAMETERS = list(itertools.product(*[[10, 50, 100, 150], [20, 50, 100, 150, 250], [1], [1], [0.5], [1]]))

ANT_PARAMETERS = list(itertools.product(*[[10, 50, 100, 150], [20, 50, 100, 150, 250], [3], [1], [0.5], [1]]))
TEST_ITERATIONS = 5


def main():
    for file in glob.glob(dir_path_local, recursive=True):
        resultName = file.split("\\")[-1].split(".")[0]
        metaData, nodes, demand = parseRawData(file)
        ant = Ant(metaData, nodes, demand, 0, 0, 0, 0, 0, 0, testIterCount=TEST_ITERATIONS)

        # resultFileName = f"../Data/ParsedData/Ant/{resultName}_{ant.alpha}_{ant.beta}.xml"
        resultFileName = f"../Data/ParsedData/Ant/{resultName}_3_1.xml"

        resultsFile = open(resultFileName, "a+")
        resultsFile.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        resultsFile.write('<results>\n')
        resultsFile.close()

        print("")
        print(f"{resultName}.xml")
        start_time = datetime.now().strftime("%H:%M:%S")

        index = 0
        for args in ANT_PARAMETERS:
            if args[2] == 3 and args[3] == 3:
                continue
            index += 1
            ant.setParameters(*args)
            # print(f"{index}/{len(ANT_PARAMETERS) - 16}    {args[2]}  {args[3]}    {resultName}.vrp")
            print(
                f"{index}/{len(ANT_PARAMETERS)}    {ant.alpha} {ant.beta}   {ant.nAnt} {ant.nIter}    {resultName}.vrp")

            now = datetime.now()
            test_time = now.strftime("%H:%M:%S")
            print(f"Start time: {start_time}  |  {test_time}")
            time = timeit.timeit(ant.run, number=1) / TEST_ITERATIONS

            res = ant.returnBest()
            resultsFile = open(resultFileName, "a+")
            resultsFile.write(
                f'<test alpha="{ant.alpha}" beta="{ant.beta}" antCount="{ant.nAnt}" '
                f'iterationCount="{ant.nIter}" avgTime="{time}">\n')
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
        folder = "Genetic" if ALGTYPE.GENETIC == mode else "Ant"
        i = 0
        for _ in glob.glob(f"..\\Results\\{folder}\\{resultName}_graph.jpg"):
            i += 1
        if i != 0:
            resultName += f"_({i})"
        result.bestRouteGraph(f"..\\Results\\{folder}\\{resultName}_graph.jpg")
        result.timeGraph(path=f"..\\Results\\{folder}\\{resultName}_time.jpg")
        result.scoreGraph(path=f"..\\Results\\{folder}\\{resultName}_score.jpg")
        result.lostCapacityGraph(path=f"..\\Results\\{folder}\\{resultName}_capacityLost.jpg")



if __name__ == "__main__":
    main()

    # analyseData(ALGTYPE.ANT)
