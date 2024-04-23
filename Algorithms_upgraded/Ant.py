import timeit
import numpy as np
from Algorithms_upgraded.Algorithm import Algorithm, Routes
from Data.DataParser import parseRawData


class Ant(Algorithm):
    nAnt: int
    nIter: int
    alpha: float
    beta: float
    evapRate: float
    Q: float
    testIterCount: int

    def __init__(self,metadata, nodes, demand, nAnt, nIter, a, b, evapRate, Q, testIterCount=5):
        super().__init__(metadata, nodes, demand)
        self.nAnt=nAnt
        self.nIter=nIter
        self.alpha=a
        self.beta=b
        self.evapRate=evapRate
        self.Q=Q
        self.testIterCount=testIterCount

    def run(self):
        self.bestRoutes = []
        counter = 0
        while counter < self.testIterCount:
            counter += 1
            print(f"Running test {counter}")
            pherMatrix = np.ones((self.metaData['DIMENSION'], self.metaData['DIMENSION']))
            it=0
            while it < self.nIter:
                it+=1
                routeList=[]
                for ant in range(self.nAnt):
                    visited = [False] * self.metaData['DIMENSION']
                    currentNode = 0
                    visited[currentNode] = True
                    route = [currentNode]
                    cap = 0
                    score = 0
                    while False in visited:
                        unvisited = np.where(np.logical_not(visited))[0]
                        prob =np.array([(pherMatrix[currentNode, unvisitedPoint] ** self.alpha /
                                       self.distMatrix[currentNode][unvisitedPoint] ** self.beta) for unvisitedPoint in unvisited])
                        prob /= np.sum(prob)
                        nextNode = np.random.choice(unvisited, p=prob)
                        if (cap + self.demand[nextNode]) > self.metaData['CAPACITY']:
                            route.append(0)
                            cap = self.demand[nextNode]
                            score += (self.distMatrix[currentNode][0]+self.distMatrix[0][nextNode])
                        else:
                            cap += self.demand[nextNode]
                            score += self.distMatrix[currentNode][nextNode]
                        route.append(nextNode)
                        visited[nextNode] = True
                        currentNode = nextNode
                    score += self.distMatrix[0][route[-1]]
                    route.append(0)
                    routeList.append(Routes(route,score))
                pherMatrix *= self.evapRate
                for route in routeList:
                    for i in range(self.metaData['DIMENSION']-1):
                        pherMatrix[route.route[i], route.route[i + 1]] += self.Q / route.score
            routeList.sort(key = lambda x: x.score)
            self.bestRoutes.append(routeList[0])

    def returnBest(self):
        return self.bestRoutes

    def setParameters(self, *args):
        self.nAnt = args[0]
        self.nIter = args[1]
        self.alpha = args[2]
        self.beta = args[3]
        self.evapRate = args[4]
        self.Q = args[5]


def main():
    metaData, nodes, demand = parseRawData("../Data/RawData/christofides/CMT4.vrp")
    alg = Ant(metaData, nodes, demand, 20, 100, 1, 1, 0.6, 1)
    print(timeit.timeit(alg.run, number=1))
    s=alg.returnBest()
    print([a.score for a in s])


if __name__=="__main__":
    main()