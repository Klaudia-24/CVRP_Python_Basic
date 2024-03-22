import math
import numpy as np
import random
from Data.Extractor import Extract
import matplotlib.pyplot as plt
from typing import List
from scipy.spatial import distance_matrix
from GenecticAlg import GeneticAlgorithm
import timeit


# f"../cmtResultGen/resultsCMT{fileIndex}.xml"
# f"../Data/CMT/CMT{fileIndex}.vrp"
# print(f"resultsCMT{fileIndex}.xml   Running test for cOpt={cOpt} mOpt={mOpt}  pOpt={pOpt}  gOpt={gOpt}")

# def main2():
#     data = Extract("../Data/CMT/CMT1.vrp")
#     demand = [data['nodes'][i]['demand'] for i in data['nodes'].keys()]
#     nodes = np.empty(shape=[0, 2])
#     for elem in data['nodes'].keys():
#         nodes = np.append(nodes, [[data['nodes'][elem]['x'], data['nodes'][elem]['y']]], axis=0)
#         # __init__(self,crossChance, mutationChance, populationSize, genCount,capacity,demand, nodesDistance)
#     alg = GeneticAlgorithm(0.15, 0.05, 200, 200, data['capacity'], demand, distance_matrix(nodes, nodes))
#     alg.run(1, 1, 2)
#     print(alg.returnBest().score)
#     print(alg.returnBest().route)


def main():
    # crossOptions = (0.05, 0.25)
    # mutationOptions = (0.01, 0.2)
    # populationOptions = (5, 10, 20, 40, 80, 160)
    # generationOptions = (10, 20, 40, 80, 160, 320)

    crossOptions = (0.05, 0.25)
    mutationOptions = (0.01, 0.2)
    populationOptions = (5, 10, 20, 40, 80, 160)
    generationOptions = (10, 20, 40, 80, 160, 320)
    # files = (1, 2, 3, 4, 5, 11, 12)
    files = (1, 12)
    for fileIndex in files:
        resultFileName = f"../cmtResultGen/result_1_2/resultsCMT{fileIndex}_test_11.xml"

        file = open(resultFileName, "a+")
        file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        file.write('<results>\n')
        file.close()
        print("")
        print(f"resultsCMT{fileIndex}.xml")
        # f"../Data/CMT/CMT{fileIndex}.vrp"
        # data = Extract("../Data/CMT/CMT1.vrp")
        data = Extract(f"../Data/CMT/CMT{fileIndex}.vrp")
        demand = [data['nodes'][i]['demand'] for i in data['nodes'].keys()]
        nodes = np.empty(shape=[0, 2])
        for elem in data['nodes'].keys():
            nodes = np.append(nodes, [[data['nodes'][elem]['x'], data['nodes'][elem]['y']]], axis=0)
        counter = 0
        total = len(crossOptions)*len(mutationOptions)*len(populationOptions)*len(generationOptions)
        for cOpt in crossOptions:
            for mOpt in mutationOptions:
                print(f"cOpt={cOpt} mOpt={mOpt}")
                for pOpt in populationOptions:
                    print("")
                    for gOpt in generationOptions:
                        counter += 1
                        print(f"{counter}/{total}  resultsCMT{fileIndex}.xml    Running test for cOpt={cOpt} mOpt={mOpt}  pOpt={pOpt}  gOpt={gOpt}")
                        file = open(resultFileName, "a+")
                        file.write(f'<test crossOption="{cOpt}" mutationOption="{mOpt}" population="{pOpt}" generationCount="{gOpt}">\n')
                        for i in range(5):
                            print(f"test {i+1}")
                            algorithm = GeneticAlgorithm(cOpt, mOpt, pOpt, gOpt, data['capacity'], demand, nodes)
                            time = timeit.timeit(algorithm.run, number=1)
                            file.write(f'<route id="{i}" score="{algorithm.returnBestScore()}" time="{time}">\n')
                            file.write(str(algorithm.returnBestRoute()) + "\n")
                            file.write('</route>\n')
                        file.write('</test>\n')
                        file.close()
        file = open(resultFileName, "a+")
        file.write('</results>\n')
        file.close()


if __name__ == "__main__":
    main()

# def main():
#     crossOptions = (0.05, 0.1, 0.15, 0.2, 0.25)
#     mutationOptions = (0.01, 0.02, 0.05, 0.1, 0.2)
#     populationOptions = (5, 10, 20, 40, 80)
#     generatonOptions = (10, 20, 40, 80, 160)
#     files=(5,11,12)
#     for fileIndex in files:
#         file=open(f"../cmtResult/result_1_2/resultsCMT{fileIndex}.xml","a+")
#         file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
#         file.write('<results>\n')
#         file.close()
#         print("")
#         for cOpt in crossOptions:
#             for mOpt in mutationOptions:
#                 for pOpt in populationOptions:
#                     for gOpt in generatonOptions:
#                         print(f"resultsCMT{fileIndex}.xml   Running test for cOpt={cOpt} mOpt={mOpt}  pOpt={pOpt}  gOpt={gOpt}")
#                         file = open(f"../cmtResult/result_1_2/resultsCMT{fileIndex}.xml", "a+")
#                         file.write(f'<test crossOption="{cOpt}" mutationOption="{mOpt}" population="{pOpt}" generationCount="{gOpt}">\n')
#                         for i in range(5):
#                             algorithm=GeneticAlgorithm(f"../Data/CMT/CMT{fileIndex}.vrp",cOpt,mOpt,pOpt,gOpt,Extract)
#                             algorithm.run(1, 2, 5)
#                             file.write(f'<route id="{i}" score="{algorithm.returnBest().getScore()}">\n')
#                             file.write(algorithm.returnBest().strp()+"\n")
#                             file.write('</route>\n')
#                         file.write('</test>\n')
#                         file.close()
#         file = open(f"../cmtResult/result_1_2/resultsCMT{fileIndex}.xml", "a+")
#         file.write('</results>\n')
#         file.close()
