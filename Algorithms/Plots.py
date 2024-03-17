from Data.Extractor import Extract
import xmltodict
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import itertools


def distance(node1, node2):
    return np.sqrt(np.sum((node1 - node2) ** 2))


class Route:
    id: int
    score: float
    time: float
    route: str

    def __init__(self, id, score, time, nodeArrays, route):
        self.id = id
        self.score = score
        self.routeArrays = nodeArrays
        self.time = time
        self.k_Opt()
        self.route = route
        self.bestScore = self.recalculateScore()

    def plotRoute(self, path):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        for i, data in enumerate(self.routeArrays):
            ax.plot(*data.T, marker='.', label=f"Trasa {i+1}")
        plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
        plt.tight_layout()
        plt.pause(0.1)
        fig.savefig(path)
        plt.close(fig)

    def calculate(self, route):
        return sum([distance(route[i - 1], route[i]) for i in range(1, len(route))])

    def recalculateScore(self):
        return sum([self.calculate(x) for x in self.routeArrays])

    def routeCount(self):
        return len(self.routeArrays)

    def k_Opt(self):
        for x in range(len(self.routeArrays)):
            improved = True
            size = len(self.routeArrays[x])
            while improved:
                improved = False
                for i in range(1, size - 2):
                    for j in range(i + 1, size):
                        if j - i == 1: continue
                        newRoute = self.routeArrays[x][:].copy()
                        newRoute[i:j] = newRoute[j - 1:i - 1:-1]
                        if self.calculate(newRoute) < self.calculate(self.routeArrays[x]):
                            improved = True
                            self.routeArrays[x] = newRoute[:]


class DataAlgGen:
    cross: float
    mut: float
    popC: int
    genC: int
    routes: List

    def __init__(self, data, nodes):
        self.routes = list()
        self.cross = data['@crossOption']
        self.mut = data['@mutationOption']
        self.popC = data['@population']
        self.genC = data['@generationCount']
        self.carCapacity = nodes['capacity']
        self.ar = np.empty(shape=[0, 2])
        self.demands = [nodes['nodes'][x]['demand'] for x in nodes['nodes'].keys()]
        for elem in nodes['nodes'].keys():
            self.ar = np.append(self.ar, [[nodes['nodes'][elem]['x'], nodes['nodes'][elem]['y']]], axis=0)
        print(f"parsing {self.cross}, {self.mut}, {self.popC}, {self.genC}")
        for ob in data['route']:
            self.routes.append(
                Route(int(ob['@id']), float(ob['@score']), float(ob['@time']), self.parseRouteArrays(ob['#text']),
                      ob['#text']))

    def parseRouteArrays(self, fullRoute):
        result = []
        routeArray = np.empty(shape=[0, 2])
        routeArray = np.append(routeArray, [self.ar[0]], axis=0)
        for node in fullRoute.replace("]", "").replace("[", "").split(",")[1:]:
            if int(node) == 1:
                routeArray = np.append(routeArray, [self.ar[int(node) - 1]], axis=0)
                result.append(routeArray)
                routeArray = np.empty(shape=[0, 2])
                routeArray = np.append(routeArray, [self.ar[0]], axis=0)
            else:
                routeArray = np.append(routeArray, [self.ar[int(node) - 1]], axis=0)
        return result

    def plotBestResult(self, path):
        self.routes.sort(key=lambda x: x.bestScore)
        self.routes[0].plotRoute(path)

    def bestResult(self):
        return self.routes[0]

    def showScores(self):
        self.routes.sort(key=lambda x: x.bestScore)
        return f"{self.routes[0].bestScore} {self.routes[0].score}"

    def avgScore(self):
        scores = sum([x.bestScore for x in self.routes])
        return scores / len(self.routes)

    def avgTime(self):
        return sum([x.time for x in self.routes]) / len(self.routes)

    def minScore(self):
        return min([x.bestScore for x in self.routes])

    def maxScore(self):
        return max([x.bestScore for x in self.routes])

    def lostCapacityRate(self):
        result = len(self.routes) * sum(self.demands)
        routeCount = 0
        for x in range(len(self.routes)):
            routeCount += self.routes[x].routeCount()
        return result / (routeCount * self.carCapacity)

    def __str__(self):
        return f"{self.carCapacity},{self.cross},{self.mut},{self.popC},{self.genC},{self.avgScore()},{self.minScore()},{self.maxScore()},{self.lostCapacityRate()}, {self.avgTime()}\n"


class DataAnt:
    nAnt: int
    nInter: int
    routes: List

    def __init__(self, data, nodes):
        self.routes = list()
        self.nAnt = data['@nAnt']
        self.nInter = data['@nInter']
        self.carCapacity = nodes['capacity']
        self.ar = np.empty(shape=[0, 2])
        self.demands = [nodes['nodes'][x]['demand'] for x in nodes['nodes'].keys()]
        for elem in nodes['nodes'].keys():
            self.ar = np.append(self.ar, [[nodes['nodes'][elem]['x'], nodes['nodes'][elem]['y']]], axis=0)
        print(f"parsing {self.nAnt}, {self.nInter}")
        for ob in data['route']:
            self.routes.append(
                Route(int(ob['@id']), float(ob['@score']), float(ob['@time']), self.parseRouteArrays(ob['#text']),
                      ob['#text']))

    def parseRouteArrays(self, fullRoute):
        result = []
        routeArray = np.empty(shape=[0, 2])
        routeArray = np.append(routeArray, [self.ar[0]], axis=0)
        for node in fullRoute.replace("]", "").replace("[", "").split(",")[1:]:
            if int(node) == 1:
                routeArray = np.append(routeArray, [self.ar[int(node) - 1]], axis=0)
                result.append(routeArray)
                routeArray = np.empty(shape=[0, 2])
                routeArray = np.append(routeArray, [self.ar[0]], axis=0)
            else:
                routeArray = np.append(routeArray, [self.ar[int(node) - 1]], axis=0)
        return result

    def plotBestResult(self, path):
        self.routes.sort(key=lambda x: x.bestScore)
        self.routes[0].plotRoute(path)

    def bestResult(self):
        return self.routes[0]

    def showScores(self):
        self.routes.sort(key=lambda x: x.bestScore)
        return f"{self.routes[0].bestScore} {self.routes[0].score} {self.routes[0].score2}"

    def avgScore(self):
        scores = sum([x.bestScore for x in self.routes])
        return scores / len(self.routes)

    def avgTime(self):
        return sum([x.time for x in self.routes]) / len(self.routes)

    def minScore(self):
        return min([x.bestScore for x in self.routes])

    def maxScore(self):
        return max([x.bestScore for x in self.routes])

    def lostCapacityRate(self):
        result = len(self.routes) * sum(self.demands)
        routeCount = 0
        for x in range(len(self.routes)):
            routeCount += self.routes[x].routeCount()
        return result / (routeCount * self.carCapacity)

    def __str__(self):
        return f"{self.carCapacity},{self.nAnt},{self.nInter},{self.avgScore()},{self.minScore()},{self.maxScore()},{self.lostCapacityRate()}, {self.avgTime()}\n"


# def main():
#     with open("Results/result_1_1/resultsCMT1.xml", "r+") as f:
#         dic = xmltodict.parse(f.read())
#         data = Extract("Data/christofides/CMT1.vrp")
#         file = open("Results/parsedResults/resultsCMT1.txt", "w")
#         i = 1
#         best = Route(0, np.inf, 0, [], "")
#         for x in dic['results']['test']:
#             z = DataAlgGen(x, data)
#             z.plotBestResult(f"Results/plots/AG3/plot{i}.jpg")
#             if z.bestResult().score < best.score:
#                 best = z.bestResult()
#             file.write(z.__str__())
#             i += 1
#         with open("Results/result_1_1/bestCMT1.xml", "w") as g:
#             g.write(f"{best.score} {best.route}")
#         best.plotRoute("Results/result_1_1/bestCMT1.jpg")
#         file.close()



# ************ ******* ** ANT PLOT ******* ******* *******
# def main():
#     cmtFile = "CMT12"
#
#     with open(f"../cmtResultAnt/results{cmtFile}.xml", "r+") as f:
#         dic = xmltodict.parse(f.read())
#         data = Extract(f"../Data/CMT/{cmtFile}.vrp")
#         file = open(f"../cmtResultAnt/results{cmtFile}.txt", "w")
#         i = 1
#         # for x in dic['results']['test']:
#         #     z = DataAnt(x, data)
#         #     nAnt = x['@nAnt']
#         #     nInter = x['@nInter']
#         #     z.plotBestResult(f"../cmtPlotsAnt/{cmtFile}/Routs/plot_{nAnt}_{nInter}.jpg")
#         #     file.write(z.__str__())
#         #     i += 1
#         best = Route(0, np.inf, 0, [], "")
#         for x in dic['results']['test']:
#             z = DataAnt(x, data)
#             nAnt = x['@nAnt']
#             nInter = x['@nInter']
#             x, y = 0, 0
#             # z.plotBestResult(f"../cmtPlotsAnt/{cmtFile}/Routs/plot_{nAnt}_{nInter}.jpg")
#             if z.bestResult().score < best.score:
#                 best = z.bestResult()
#                 x = z.nAnt
#                 y = z.nInter
#             file.write(z.__str__())
#             i += 1
#         with open(f"../cmtResultAnt/bestResult{cmtFile}.txt", "w") as g:
#             g.write(f"{x} {y} {best.score} \n {best.route}")
#         best.plotRoute(f"../cmtPlotsAnt/{cmtFile}/bestRoutePlot.jpg")
#         file.close()


# ************ ******* ** GENETIC PLOT ******* ******* *******
def main():
    cmtFile = "CMT12"
    c = 1
    m = 2

    with open(f"../cmtResultGen/result_{c}_{m}/results{cmtFile}.xml", "r+") as f:
        dic = xmltodict.parse(f.read())
        data = Extract(f"../Data/CMT/{cmtFile}.vrp")
        file = open(f"../cmtResultGen/result_{c}_{m}/results{cmtFile}.txt", "w")
        i = 1
        best = Route(0, np.inf, 0, [], "")
        y = []
        for x in dic['results']['test']:
            z = DataAlgGen(x, data)
            cross = x['@crossOption']
            mutation = x['@mutationOption']
            population = x['@population']
            genCount = x['@generationCount']
            # z.plotBestResult(f"../cmtPlotsGen/result_1_1/{cmtFile}/Routs/plot_{cross}_{mutation}_{population}_{genCount}.jpg")
            if z.bestResult().score < best.score:
                best = z.bestResult()
                y = [z.cross, z.mut, z.popC, z.genC]
            file.write(z.__str__())
            i += 1
        with open(f"../cmtResultGen/result_{c}_{m}/bestResult{cmtFile}.txt", "w") as g:
            g.write(f"{y} {best.score} \n {best.route}")
        best.plotRoute(f"../cmtPlotsGen/result_{c}_{m}/{cmtFile}/bestRoutePlot.jpg")
        file.close()


if __name__ == "__main__":
    main()
