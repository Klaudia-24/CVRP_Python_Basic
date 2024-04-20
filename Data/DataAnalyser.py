from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import glob

import matplotlib

from Data.DataParser import parseRawData
import xmltodict
import ast
import numpy as np
import matplotlib.pyplot as plt
from Algorithms_upgraded.Algorithm import distMatrix
import itertools

DIR_PATH_LOCAL = ".\\RawData\\**\\"


class ALGTYPE(Enum):
    GENETIC = 1
    ANT = 2


@dataclass
class Solutions(ABC):
    avgTime: float
    score: float
    rawRoute: list[int]
    subRoutes: list[list[int]] = field(init=False)

    @abstractmethod
    def param(self):
        pass


@dataclass
class GeneticSolutions(Solutions):
    crossMode: int
    mutMode: int
    crossRatio: float
    mutRatio: float
    popCount: int
    genCount: int

    def param(self):
        return self.crossRatio, self.mutRatio, self.popCount

    def __repr__(self):
        return ("crossMode={}, mutMode={}, avgTime={}, score={}, crossRatio={}, mutRatio={}, popCount={}, genCount={}"
                .format(self.crossMode, self.mutMode, self.avgTime, self.score, self.crossRatio, self.mutRatio,
                        self.popCount, self.genCount))


@dataclass
class AntSolutions:
    alpha: float
    beta: float
    antCount: int
    iterCount: int


class Data:
    def __init__(self, metadata, nodes, demand, parsedData: list[Solutions]):
        self.metaData = metadata
        self.nodes = nodes
        self.distance = distMatrix(nodes)
        self.demand = demand
        self.parsedData = parsedData
        self.splitRoutes()
        self.k_Opt()
        self.parsedData.sort(key=lambda x: x.score)

    def splitRoutes(self):
        for x in self.parsedData:
            subRoute = [0]
            cap = 0
            x.subRoutes = []
            for node in x.rawRoute:
                if cap + self.demand[node] > self.metaData['CAPACITY']:
                    subRoute.append(0)
                    x.subRoutes.append(subRoute)
                    subRoute = [0, node]
                    cap = self.demand[node]
                else:
                    cap += self.demand[node]
                    subRoute.append(node)
            subRoute.append(0)
            x.subRoutes.append(subRoute)

    def k_Opt(self):
        for x in self.parsedData:
            for sub in range(len(x.subRoutes)):
                improved = True
                size = len(x.subRoutes[sub])
                while improved:
                    improved = False
                    for i in range(1, size - 2):
                        for j in range(i + 1, size):
                            if j - i == 1:
                                continue
                            newRoute = x.subRoutes[sub][:].copy()
                            newRoute[i:j] = newRoute[j - 1:i - 1:-1]
                            if self.fitness(newRoute) < self.fitness(x.subRoutes[sub]):
                                improved = True
                                x.subRoutes[sub] = newRoute[:]
            x.score = self.fitness(list(itertools.chain.from_iterable(x.subRoutes)))
            x.rawRoute = list(filter(lambda a: a != 0, list(itertools.chain.from_iterable(x.subRoutes))))

    def fitness(self, route: list[int]) -> float:
        return sum([self.distance[route[i - 1]][route[i]] for i in range(1, len(route))])

    def bestRoute(self):
        return self.parsedData[0]

    def bestRouteGraph(self, path=None):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        for i, subRoute in enumerate(self.parsedData[0].subRoutes, start=1):
            ax.plot(*np.array([self.nodes[i] for i in subRoute]).T, marker='.', label=f"Trasa {i}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
        plt.tight_layout()
        plt.pause(0.1)
        if path is None:
            plt.show()
        else:
            fig.savefig(path)
            plt.close(fig)

    def timeGraph(self, par=None, path=None):
        if par is None:
            par = self.bestRoute().param()
        fig, ax = plt.subplots(nrows=1, ncols=1)
        if isinstance(self.bestRoute(), GeneticSolutions):
            popCounts = list(set([x.popCount for x in self.parsedData]))
            popCounts.sort()
            for pop in popCounts:
                points = list(set([(x.genCount, x.avgTime) for x in self.parsedData if x.crossRatio==par[0] and x.mutRatio == par[1] and x.popCount == pop]))
                points.sort(key= lambda a: a[0])
                points=np.array(points)
                # ax.plot(points[:,0],points[:,1], marker=".", label=f"c={par[0]*100}% m={par[1]*100}% p={pop}")
                ax.plot(points[:, 0], points[:, 1], marker=".", label=f"p={pop}")
            plt.xlabel("Liczba generacji")
            plt.ylabel("Czas [s]")
            plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
            plt.tight_layout()
            plt.pause(0.1)
            if path is None:
                plt.show()
            else:
                fig.savefig(path)
                plt.close(fig)

    def timeGraphFor(self, par=None, path=None):
        if par is None:
            par = self.bestRoute().param()
        fig, ax = plt.subplots(nrows=1, ncols=1)
        if isinstance(self.bestRoute(), GeneticSolutions):
            points = list(set([(x.genCount, x.avgTime) for x in self.parsedData if
                               x.crossRatio == par[0] and x.mutRatio == par[1] and x.popCount == par[2]]))
            points.sort(key=lambda a: a[0])
            points = np.array(points)
            ax.plot(points[:, 0], points[:, 1], marker=".", label=f"c={par[0]*100}% m={par[1]*100}% p={par[2]}")
            # ax.plot(points[:, 0], points[:, 1], marker=".", label=f"p={par[2]}")
            plt.xticks(ticks=points[:, 0])
            # for i in points:
            #     ax.text(i[0], i[1] + 0.15, f"{np.round(i[1], 2)}", ha="center")
            plt.xlabel("Liczba generacji")
            plt.ylabel("Czas [s]")
            plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
            plt.tight_layout()
            plt.pause(0.1)
            if path is None:
                plt.show()
            else:
                fig.savefig(path)
                plt.close(fig)

    def scoreGraph(self, par=None, path=None):
        if par is None:
            par = self.bestRoute().param()
        fig, ax = plt.subplots(nrows=1, ncols=1)
        if isinstance(self.bestRoute(), GeneticSolutions):
            generations = list(set([x.genCount for x in self.parsedData if
                                    x.crossRatio == par[0] and x.mutRatio == par[1] and x.popCount == par[2]]))
            generations.sort()
            min_list = [min([x.score for x in self.parsedData if
                             x.crossRatio == par[0] and x.mutRatio == par[1] and
                             x.popCount == par[2] and x.genCount == y]) for y in generations]
            max_list = [max([x.score for x in self.parsedData if
                             x.crossRatio == par[0] and x.mutRatio == par[1] and
                             x.popCount == par[2] and x.genCount == y]) for y in generations]

            avg_list = [np.average([x.score for x in self.parsedData if
                                    x.crossRatio == par[0] and x.mutRatio == par[1] and
                                    x.popCount == par[2] and x.genCount == y]) for y in generations]

            ax.plot(generations, min_list, marker=".", label=f"Minimum")
            ax.plot(generations, avg_list, marker=".", label=f"Średnia")
            ax.plot(generations, max_list, marker=".", label=f"Maksimum")

            # plt.xticks(ticks=generations)
            plt.xlabel("Liczba generacji")
            plt.ylabel("Dystans")
            # plt.grid(linestyle='--')
            plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
            plt.tight_layout()
            plt.pause(0.1)
            if path is None:
                plt.show()
            else:
                fig.savefig(path)
                plt.close(fig)

    # def _lostCapacity(self, list):
    #     result = 0
    #     for element in list:
    #         result += sum([self.metaData['CAPACITY'] - sum([self.demand[node] for node in subroute]) for subroute in
    #                        element.subRoutes])
    #     return result / len(list)

    def _lostCapacity(self, list):
        result = 0
        for element in list:
            result += sum(
                [(self.metaData['CAPACITY'] - sum([self.demand[node] for node in subroute])) / self.metaData["CAPACITY"]
                 for subroute in element.subRoutes]) / len(element.subRoutes)
        return 100 * result / len(list)
    def lostCapacityGraph(self, par=None, path=None):
        if par is None:
            par = self.bestRoute().param()
        fig, ax = plt.subplots(nrows=1, ncols=1)
        if isinstance(self.bestRoute(), GeneticSolutions):
            popCount = list(set([x.popCount for x in self.parsedData]))
            popCount.sort()
            genCount = list(set([x.genCount for x in self.parsedData]))
            genCount.sort()
            spacer = -len(genCount) // 2
            xaxis = [x for x in range(10, len(genCount) * 10 + 1, 10)]
            for pop in popCount:
                capLoss = [self._lostCapacity(
                    [sol for sol in self.parsedData if sol.crossRatio == par[0] and sol.mutRatio == par[1] and
                     sol.popCount == pop and sol.genCount == x]) for x in genCount]
                ax.bar([x + spacer for x in xaxis], capLoss, label=f"{pop} osobników", width=1)
                spacer += 1
            plt.xticks(ticks=xaxis, labels=[str(x) for x in genCount])
            plt.legend(bbox_to_anchor=(0.5, 1.3), loc='upper center')
            plt.ylabel("Pojemność [%]")
            plt.xlabel("Liczba generacji")
            plt.tight_layout()
            plt.pause(0.1)
            if path is None:
                plt.show()
            else:
                fig.savefig(path)
                plt.close(fig)

def subRouteToRouteGraph(dataList, path=None):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    if isinstance(dataList[0].bestRoute(), GeneticSolutions):
        for dataset in dataList:
            points = [[x.score, max([dataset.fitness(subroute) for subroute in x.subRoutes])] for x in
                      dataset.parsedData]
            points = np.array(points)
            ax.scatter(points[:, 0], points[:, 1], marker=".",
                       label=f"K={dataset.parsedData[0].crossMode} M={dataset.parsedData[0].mutMode}")
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    plt.ylabel("Trasa")
    plt.xlabel("Całowita długość")
    plt.tight_layout()
    plt.pause(0.1)
    if path is None:
        plt.show()
    else:
        fig.savefig(path)
        plt.close(fig)


def parseData(filePath, DIR_PATH=DIR_PATH_LOCAL, mode: ALGTYPE = ALGTYPE.GENETIC):
    """Parse generated data from tests."""
    rawDataFileName = filePath.split("\\")[-1].split("_")[0][6:]
    cross = int(filePath.split("\\")[-1].split("_")[1])
    mut = int(filePath.split("\\")[-1].split("_")[2][0])
    metaData, nodes, demand = parseRawData(glob.glob(f"{DIR_PATH}{rawDataFileName}.vrp", recursive=True)[0])
    with open(filePath, "r") as f:
        rawDataDict = xmltodict.parse(f.read())
    res = []
    if mode == ALGTYPE.GENETIC:
        # rawDataDict['results']['test'] = [rawDataDict['results']['test']]# for only one test
        res = [
            GeneticSolutions(float(x['@avgTime']), float(path['@score']), ast.literal_eval(path['#text']), cross, mut,
                             float(x['@crossOption']), float(x['@mutationOption']), int(x['@population']),
                             int(x['@generationCount'])) for x in rawDataDict['results']['test'] for path in x['route']]
    elif mode == ALGTYPE.ANT:
        pass
    return Data(metaData, nodes, demand, res)


if __name__ == "__main__":

    # christofides resultCMT1 ; resultCMT12

    # results = [parseData("..\\cmtResultGen\\iter_20\\resultCMT1_1_2.xml", mode=ALGTYPE.GENETIC)]
    # results.append(parseData("..\\cmtResultGen\\iter_20\\resultCMT1_1_1.xml", mode=ALGTYPE.GENETIC))
    # results.append(parseData("..\\cmtResultGen\\iter_20\\resultCMT1_2_2.xml", mode=ALGTYPE.GENETIC))
    # results.append(parseData("..\\cmtResultGen\\iter_20\\resultCMT1_2_1.xml", mode=ALGTYPE.GENETIC))

    # results = [parseData(".\\ParsedData\\Genetic\\resultCMT1_1_2.xml", mode=ALGTYPE.GENETIC)]
    # results.append(parseData(".\\ParsedData\\Genetic\\resultCMT1_1_1.xml", mode=ALGTYPE.GENETIC))
    # results.append(parseData(".\\ParsedData\\Genetic\\resultCMT1_2_2.xml", mode=ALGTYPE.GENETIC))
    # results.append(parseData(".\\ParsedData\\Genetic\\resultCMT1_2_1.xml", mode=ALGTYPE.GENETIC))

    # augerat_A

    # results = [parseData("..\\cmtResultGen\\iter_20\\resultCMT1_1_2.xml", mode=ALGTYPE.GENETIC)]
    # results.append(parseData("..\\cmtResultGen\\iter_20\\resultCMT1_1_1.xml", mode=ALGTYPE.GENETIC))
    # results.append(parseData("..\\cmtResultGen\\iter_20\\resultCMT1_2_2.xml", mode=ALGTYPE.GENETIC))
    # results.append(parseData("..\\cmtResultGen\\iter_20\\resultCMT1_2_1.xml", mode=ALGTYPE.GENETIC))

    results = [parseData(".\\ParsedData\\Genetic\\resultCMT1_1_2.xml", mode=ALGTYPE.GENETIC)]
    results.append(parseData(".\\ParsedData\\Genetic\\resultCMT1_1_1.xml", mode=ALGTYPE.GENETIC))
    results.append(parseData(".\\ParsedData\\Genetic\\resultCMT1_2_2.xml", mode=ALGTYPE.GENETIC))
    results.append(parseData(".\\ParsedData\\Genetic\\resultCMT1_2_1.xml", mode=ALGTYPE.GENETIC))

    # result = parseData(".\\ParsedData\\Genetic\\resultCMT1_1_2.xml", mode=ALGTYPE.GENETIC)
    # print(result.bestRoute())
    # result.bestRouteGraph(".\\Results\\Genetic\\fig1.jpg")
    # result.scoreGraph(par=(0.05, 0.2, 160))
    # result.lostCapacityGraph()
    # subRouteToRouteGraph(results, path=f"..\\Results\\Genetic\\resultCMT12_routsBest.jpg") # uncomment line in parseData fun for one test
    subRouteToRouteGraph(results, path=f"..\\Results\\Genetic\\resultCMT1_routsAll.jpg")
