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

DIR_PATH_LOCAL = "..\\RawData\\**\\"


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

    def timeGraph(self, path=None):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.set_prop_cycle('color', matplotlib.colormaps['rainbow'].resampled(26)(np.linspace(0, 1, 26)))
        if isinstance(self.bestRoute(), GeneticSolutions):
            parameters = list(itertools.product(
                *[set([x.crossRatio for x in self.parsedData]), set([x.mutRatio for x in self.parsedData]),
                  set([x.popCount for x in self.parsedData])]))
            for par in parameters:
                points = list(set([(x.genCount, x.avgTime) for x in self.parsedData if
                                   x.crossRatio == par[0] and x.mutRatio == par[1] and x.popCount == par[2]]))
                points.sort(key=lambda a: a[0])
                points = np.array(points)
                ax.plot(points[:, 0], points[:, 1], marker=".", label=f"c={par[0]} m={par[1]} p={par[2]}")
            plt.xlabel("Generacje")
            plt.ylabel("Czas [sekundy]")
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
            ax.plot(points[:, 0], points[:, 1], marker=".", label=f"c={par[0]} m={par[1]} p={par[2]}")
            plt.xticks(ticks=points[:, 0])
            for i in points:
                ax.text(i[0], i[1] + 0.15, f"{np.round(i[1], 2)}", ha="center")
            plt.xlabel("Generacje")
            plt.ylabel("Czas [sekundy]")
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

            ax.plot(generations, min_list, marker=".", label=f"minimum")
            ax.plot(generations, max_list, marker=".", label=f"maximum")
            ax.plot(generations, avg_list, marker=".", label=f"average")
            # plt.xticks(ticks=generations)
            plt.xlabel("Generacje")
            plt.ylabel("Koszt")
            plt.grid(linestyle='--')
            plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
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
        res = [
            GeneticSolutions(float(x['@avgTime']), float(path['@score']), ast.literal_eval(path['#text']), cross, mut,
                             float(x['@crossOption']), float(x['@mutationOption']), int(x['@population']),
                             int(x['@generationCount'])) for x in rawDataDict['results']['test'] for path in x['route']]
    elif mode == ALGTYPE.ANT:
        pass
    return Data(metaData, nodes, demand, res)


if __name__ == "__main__":
    result = parseData(".\\ParsedData\\Genetic\\resultCMT1_1_2.xml", mode=ALGTYPE.GENETIC)
    # print(result.bestRoute())
    # result.bestRouteGraph(".\\Results\\Genetic\\fig1.jpg")
    result.scoreGraph(par=(0.05, 0.2, 160))
