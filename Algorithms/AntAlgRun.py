from Algorithms.AntAlg import AntColony
from Data.Extractor import Extract
import numpy as np
import timeit


def main():
    # antsC = (10, 20, 40, 60)
    # iterCount = (25, 50, 75, 100)
    # files = (1, 2, 3, 4, 5, 11, 12)
    antsC = (60, 5)
    iterCount = (100, 5)
    files = (12, 11)
    for fileIndex in files:
        resultFileName = f"../cmtResultAnt/resultsCMT{fileIndex}_1.xml"
        file = open(resultFileName, "a+")
        file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        file.write('<results>\n')
        file.close()

        counter = 0
        total = len(antsC) * len(iterCount)
        for nAnt in antsC:
            for nIter in iterCount:
                counter += 1
                print(f"{counter}/{total}  resultsCMT{fileIndex}.xml   Running test for nAnt={nAnt} nInter={nIter}")
                file = open(resultFileName, "a+")
                file.write(f'<test nAnt="{nAnt}" nInter="{nIter}">\n')
                for i in range(5):
                    print(f"Running test {i + 1}")
                    # def __init__(self, file, nAnt, nIter, a, b, evapRate, Q, Extract)
                    antGen = AntColony(f"../Data/CMT/CMT{fileIndex}.vrp", nAnt, nIter, 1, 1, 0.5, 1, Extract)
                    time = timeit.timeit(antGen.run, number=1)
                    file.write(f'<route id="{i}" score="{antGen.bestScore}" time="{time}">\n')
                    file.write(str(list((np.array(antGen.bestRoute) + 1))))
                    file.write("\n")
                    file.write('</route>\n')
                file.write('</test>\n')
                file.close()
        file = open(resultFileName, "a+")
        file.write('</results>\n')
        file.close()


if __name__ == "__main__":
    main()

# f"../cmtResultAnt/resultsCMT{fileIndex}.xml"
# f"../Data/CMT/CMT{fileIndex}.vrp"
# print(f"resultsCMT{fileIndex}.xml   Running test for cOpt={cOpt} mOpt={mOpt}  pOpt={pOpt}  gOpt={gOpt}")
