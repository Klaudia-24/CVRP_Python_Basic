from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class Routes:
    __slots__ = ['route', 'score']
    route: list[int]
    score: float


dist = lambda p1, p2: np.sqrt(((p1 - p2) ** 2).sum())


def eucDist(n1, n2):
    return np.sqrt(np.sum((n1 - n2) ** 2))


def distMatrix(nodes):
    return np.asarray([[dist(p1, p2) for p2 in nodes] for p1 in nodes])


class Algorithm(ABC):
    bestRoutes: list[Routes]

    def __init__(self, metaData, nodes, demand):
        self.metaData = metaData
        self.distMatrix = distMatrix(nodes)
        self.demand = demand
        self.bestRoutes = []

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def setParameters(self, *args):
        pass

    @abstractmethod
    def returnBest(self):
        pass
