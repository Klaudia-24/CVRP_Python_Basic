import timeit
import numpy as np
from Algorithms_upgraded.Algorithm import Algorithm, Routes
from Data.DataParser import parseData
from enum import Enum
from datetime import datetime


class Mutation(Enum):
    RANDOM = 1
    BESTSOLUTION = 2


class Genetic(Algorithm):
    popCount: int
    genCount: int
    crossRatio: float
    mutationRatio: float
    testIterCount: int

    def __init__(self, metadata, nodes, demand, crossRatio, mutationRatio, popCount, genCount, testIterCount=5):
        super().__init__(metadata, nodes, demand)
        self.popCount = popCount
        self.genCount = genCount
        self.crossRatio = crossRatio
        self.mutationRatio = mutationRatio
        self.testIterCount = testIterCount
        self.mutationMode = Mutation.RANDOM
        self.nCross = 1
        self._population = None

    def run(self):
        iter = 0

        # now = datetime.now()
        # current_time = now.strftime("%H:%M:%S")

        print(
            f"Start test for:  cross={self.crossRatio}  mut={self.mutationRatio}  pop={self.popCount}  gen={self.genCount}     |  ncros={self.nCross}   mode={self.mutationMode}")
        # print("Start time:", current_time)
        while iter < self.testIterCount:
            iter += 1
            print(f"Running test {iter}")
            self._population = self.generateInitialPopulation()
            self._population.sort(key=lambda a: a.score)
            index = 0
            while index < self.genCount:
                newPop = [x for _ in range(self.popCount // 2 + self.popCount % 2) for x in
                          self.nPointCross(*np.random.choice(5, 2, replace=False))]
                newPop = [self.mutation(x) for x in newPop]
                self._population = [Routes(x, self.fitness(x)) for x in newPop]
                self._population.sort(key=lambda x: x.score)
                index += 1
            self.bestRoutes.append(self._population[0])

    def setNCross(self, n):
        self.nCross = n

    def setMutation(self, mut: Mutation):
        self.mutationMode = mut

    def nPointCross(self, parent1ID, parent2ID):
        r1 = self._population[parent1ID].route.copy()
        r2 = self._population[parent2ID].route.copy()
        if np.random.rand() < self.crossRatio:
            crossPoints = np.random.choice(self.metaData["DIMENSION"], self.nCross, replace=False)
            crossPoints.sort()
            r1 = self._population[parent1ID].route[0:crossPoints[0]]
            r2 = self._population[parent2ID].route[0:crossPoints[0]]
            for i in range(1, self.nCross, 2):
                if i + 1 >= self.nCross:
                    r1 += self._population[parent1ID].route[crossPoints[i]:]
                    r2 += self._population[parent2ID].route[crossPoints[i]:]
                else:
                    r1 += self._population[parent1ID].route[crossPoints[i]:crossPoints[i + 1]]
                    r2 += self._population[parent2ID].route[crossPoints[i]:crossPoints[i + 1]]

            missingNodes1 = [x for x in self._population[parent2ID].route if x not in r1]
            missingNodes2 = [x for x in self._population[parent1ID].route if x not in r2]

            for i in range(0, self.nCross, 2):
                if i + 1 >= self.nCross:
                    r1 += missingNodes1[:]
                    r2 += missingNodes2[:]
                else:
                    r1 = r1[:crossPoints[i]] + missingNodes1[:crossPoints[i + 1] - crossPoints[i]] + r1[crossPoints[i]:]
                    r2 = r2[:crossPoints[i]] + missingNodes2[:crossPoints[i + 1] - crossPoints[i]] + r2[crossPoints[i]:]
                missingNodes1 = missingNodes1[crossPoints[i]:]
                missingNodes2 = missingNodes2[crossPoints[i]:]

        return [r1, r2]

    def mutation(self, route):
        selected = np.where(np.random.choice([0, 1], size=self.metaData["DIMENSION"] - 1,
                                             p=[1 - self.mutationRatio, self.mutationRatio]))[0]
        match self.mutationMode:
            case Mutation.RANDOM:
                swapPlaces = np.random.choice(self.metaData["DIMENSION"] - 1, len(selected), replace=False)
                for i, j in zip(selected, swapPlaces):
                    route[i], route[j] = route[j], route[i]
            case Mutation.BESTSOLUTION:
                copyRoute = route.copy()
                for i in selected:
                    sel = route.pop(route.index(copyRoute[i]))
                    route = self.insertWarehouse(route)
                    scoreCheck = min([(self.distMatrix[sel][route[i - 1]] + self.distMatrix[sel][route[i]], i) for i in
                                      range(1, len(route))])
                    route.insert(scoreCheck[1], sel)
                    route = list(filter(lambda a: a != 0, route))
        return route

    def insertWarehouse(self, route):
        route.insert(0, 0)
        cap = 0
        for i in range(1, len(route)):
            if cap + self.demand[route[i]] > self.metaData['CAPACITY']:
                route.insert(i, 0)
                cap = self.demand[route[i]]
            else:
                cap += self.demand[route[i]]
        route.append(0)
        return route

    def generateInitialPopulation(self):
        routeNodes = [x for x in range(1, len(self.demand))]
        routeNodes = [self.randomize(routeNodes) for _ in range(self.popCount)]
        return [Routes(x, self.fitness(x)) for x in routeNodes]

    def fitness(self, route):
        score = self.distMatrix[0][route[0]]
        current = route[0]
        cap = self.demand[route[0]]
        for nextNode in route[1:]:
            if cap + self.demand[nextNode] > self.metaData['CAPACITY']:
                score += self.distMatrix[current][0] + self.distMatrix[0][nextNode]
                cap = self.demand[nextNode]
            else:
                score += self.distMatrix[current][nextNode]
                cap += self.demand[nextNode]
            current = nextNode
        score += self.distMatrix[current][0]
        return score

    def setParameters(self, *args):
        self.crossRatio = args[0]
        self.mutationRatio = args[1]
        self.popCount = args[2]
        self.genCount = args[3]

    def randomize(self, routeNodes):
        np.random.shuffle(routeNodes)
        return routeNodes.copy()

    def returnBest(self):
        return self.bestRoutes


def main():
    metaData, nodes, demand = parseData("../Data/RawData/christofides/CMT1.vrp")
    alg = Genetic(metaData, nodes, demand, 0.3, 0.3, 160, 320, 10)
    alg.setMutation(Mutation.BESTSOLUTION)
    print(timeit.timeit(alg.run, number=1))
    s = alg.returnBest()
    print([a.score for a in s])


if __name__ == "__main__":
    main()
