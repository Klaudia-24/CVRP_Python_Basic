import math
import numpy as np
import random
from Data.Extractor import Extract
import matplotlib.pyplot as plt
from typing import List

dist = lambda p1, p2: np.sqrt(((p1 - p2) ** 2).sum())


def eucDist(n1, n2):
    return np.sqrt(np.sum((n1 - n2) ** 2))


class Chromosome:
    routes: List[int]
    score: float

    def __init__(self, route, score):
        self.route = route
        self.score = score


def distance(node1, node2):
    return np.sqrt(np.sum((node1 - node2) ** 2))


class GeneticAlgorithm:
    def __init__(self, crossChance, mutationChance, populationSize, genCount, capacity, demand, nodesDistance):
        self.genCount = genCount
        self.populationSize = populationSize
        self.mutationChance = mutationChance
        self.crossChance = crossChance
        self.capacity = capacity
        self.demand = demand
        self.nodesDistance = np.asarray([[dist(p1, p2) for p2 in nodesDistance] for p1 in nodesDistance])
        self.population: List[Chromosome] = self.initialize()

    def initialize(self):
        population: List[Chromosome] = []
        route = [x for x in range(1, len(self.demand))]
        for _ in range(self.populationSize):
            np.random.shuffle(route)
            temp = Chromosome(route, self.score(route))
            population.append(temp)
        population.sort(key=lambda x: x.score)
        return population

    def score(self, route):
        score = self.nodesDistance[0][route[0]]
        current = route[0]
        cap = self.demand[route[0]]
        for nextNode in route[1:]:
            if cap + self.demand[nextNode] > self.capacity:
                score += self.nodesDistance[current][0] + self.nodesDistance[0][nextNode]
                cap = self.demand[nextNode]
            else:
                score += self.nodesDistance[current][nextNode]
                cap += self.demand[nextNode]
            current = nextNode
        score += self.nodesDistance[current][0]
        return score

    # def score(self, route):
    #     routeToScore=[0]
    #     cap=0
    #     for nextNode in route:
    #         if cap + self.demand[nextNode] > self.capacity:
    #             routeToScore.append(0)
    #             routeToScore.append(nextNode)
    #             cap = self.demand[nextNode]
    #         else:
    #             routeToScore.append(nextNode)
    #             cap+=self.demand[nextNode]
    #     routeToScore.append(0)
    #     return sum([distance(self.nodesDistance[routeToScore[i-1]],self.nodesDistance[routeToScore[i]]) for i in range(1,len(routeToScore))])

    def run(self, sel=1, crossOpt=2, mutOpt=2):
        index = 0
        while index < self.genCount:
            newPop = []
            selected = self.selection(sel)
            match crossOpt:
                case 1:
                    newPop = [self.onePointCross(selected) for _ in range(self.populationSize)]
                case 2:
                    newPop = [self.twoPointCross(selected) for _ in range(self.populationSize)]
            match mutOpt:
                case 1:
                    newPop = [self.mutation(x) for x in newPop]
                case 2:
                    newPop = [self.optimalMutation(x) for x in newPop]
            self.population = [Chromosome(x, self.score(x)) for x in newPop]
            self.population.sort(key=lambda x: x.score)
            index += 1

    def selection(self, selOption):
        match selOption:
            case 1:
                return self.bestOf(5)

    def bestOf(self, n):
        return self.population[:n]

    def onePointCross(self, selected):
        parent = np.random.randint(0, len(selected))
        result = selected[parent].route

        if np.random.rand() < self.crossChance:
            i = np.random.randint(0, self.populationSize)

            # zabezpieczenie przed wybraniem tego samego rodzica
            while parent == i:
                i = np.random.randint(0, self.populationSize)

            crossPoints = np.random.randint(0, len(selected[parent].route))  # choose crossover points
            if np.random.rand() < 0.5:
                result = selected[parent].route[0:crossPoints] + [x for x in self.population[i].route if
                                                                  x not in selected[parent].route[0:crossPoints]]
            else:
                result = [x for x in self.population[i].route if
                          x not in selected[parent].route[crossPoints:]] + selected[parent].route[crossPoints:]
            # result = selected[parent].route[0:crossPoints] + [x for x in self.population[i].route if
            #                   x not in selected[parent].route[0:crossPoints]]
        return result

    def twoPointCross(self, selected):
        parent = np.random.randint(0, len(selected))
        result = selected[parent].route

        if np.random.rand() < self.crossChance:
            i = np.random.randint(0, self.populationSize)

            # zabezpieczenie przed wybraniem tego samego rodzica
            while (parent == i):
                i = np.random.randint(0, self.populationSize)

            crossPoint1 = np.random.randint(0, len(selected[parent].route))  # choose crossover points
            crossPoint2 = np.random.randint(0, len(selected[parent].route))

            while (crossPoint1 == crossPoint2):
                crossPoint2 = np.random.randint(0, len(selected[parent].route))
            crossPoints = [crossPoint1, crossPoint2]
            crossPoints.sort()
            result = selected[parent].route[0:crossPoints[0]] + selected[parent].route[crossPoints[1]:]
            missingNodes = [x for x in self.population[i].route if x not in result]

            place = crossPoints[0]

            for node in missingNodes:
                result.insert(place, node)
                place += 1
        return result

    def mutation(self, route):
        for i in range(0, len(route)):
            if np.random.rand() < self.mutationChance:
                temp = route[i]
                swapPoint = np.random.randint(0, len(route))
                route[i] = route[swapPoint]
                route[swapPoint] = temp
        return route

    def optimalMutation(self, route):
        for i in range(len(route)):
            if np.random.rand() < self.mutationChance:
                bestScore = self.score(route)
                currentPlacement = i
                tempNode = route.pop(i)
                for j in range(len(route)):
                    route.insert(j, tempNode)
                    currentScore = self.score(route)
                    if bestScore > currentScore:
                        bestScore = currentScore
                        currentPlacement = j
                    route.pop(j)
                route.insert(currentPlacement, tempNode)
        return route

    def returnBestRoute(self):
        route = [1]
        cap = 0
        for curNode in self.population[0].route:
            if cap + self.demand[curNode] > self.capacity:
                route.append(1)
                route.append(curNode + 1)
                cap = self.demand[curNode]
            else:
                route.append(curNode + 1)
                cap += self.demand[curNode]
        route.append(1)
        return route

    def returnBestScore(self):
        return self.population[0].score


def calculate(route, nodes):
    return sum([distance(nodes[route[i - 1] - 1], nodes[route[i] - 1]) for i in range(1, len(route))])


def main():
    data = Extract("../Data/CMT/CMT12.vrp")
    demand = [data['nodes'][i]['demand'] for i in data['nodes'].keys()]
    nodes = np.empty(shape=[0, 2])
    for elem in data['nodes'].keys():
        nodes = np.append(nodes, [[data['nodes'][elem]['x'], data['nodes'][elem]['y']]], axis=0)
    alg = GeneticAlgorithm(0.25, 0.2, 40, 80, data['capacity'], demand, nodes)
    alg.run(1, 1, 2)
    print(alg.returnBestRoute())
    print(alg.returnBestScore())
    print(calculate(alg.returnBestRoute(), nodes))


if __name__ == "__main__":
    main()
# class Chromosome:
#     def __init__(self,route,routeDepots,score=0):
#         self.route=route
#         self.routeDepots=routeDepots
#         self.score: float = score
#
#     def setScore(self,score: float):
#         self.score=score
#
#     def getScore(self):
#         return self.score
#     def strp(self):
#         result=""
#         for i in self.routeDepots:
#             result+=f'{i} '
#         return result
#
# class GeneticAlgorithm:
#     def __init__(self, file, crossChance, mutationChance, populationSize, genCount, function):
#         self.genCount = genCount
#         self.populationSize = populationSize
#         self.mutationChance = mutationChance
#         self.crossChance = crossChance
#         self.data : dict = function(file)
#         self.nodes = [i for i in range(2,len(self.data['nodes'].keys())+1)]
#         self.population: List[Chromosome] = self.generateInitialPopulation()
#
#
#     def generateRoute(self):
#         random.shuffle(self.nodes)
#         return self.nodes.copy()
#
#     def generateInitialPopulation(self):
#         population: List[Chromosome] = []
#         for _ in range(self.populationSize):
#             route=self.generateRoute()
#             temp=Chromosome(route, self.reinstateDepods(route))
#             temp.setScore(self.fitness(temp.routeDepots))
#             population.append(temp)
#         population.sort(key = lambda x: x.score)
#         return population
#
#     def fitness(self,route):
#         score=0
#         for i in range(1,len(route)):
#             score+=self.distance(self.data['nodes'][route[i]],self.data['nodes'][route[i-1]])
#         return score
#
#     def checkScore(self, route):
#         test=self.reinstateDepods(route)
#         return self.fitness(test)
#     def distance(self,a,b):
#         return math.sqrt(math.pow(b['x']-a['x'],2)+math.pow(b['y']-a['y'],2)) #bez zaokrąglania
#
#     def run(self, mode, mode2, *arg):
#         index=1
#         while index<self.genCount:
#             newPopulation=self.newPopulation(mode,arg[0])
#             if mode2==1:
#                 newPopulation=self.mutation(newPopulation)
#             if mode2==2:
#                 newPopulation = self.mutation2Try(newPopulation)
#             self.population=[]
#             for route in newPopulation:
#                 temp=Chromosome(route,self.reinstateDepods(route))
#                 temp.setScore(self.fitness(temp.routeDepots))
#                 self.population.append(temp)
#             self.population.sort(key=lambda x: x.score)
#             index+=1
#
#     def newPopulation(self, mode,*arg):
#         """1 for n best"""
#         newPop=[]
#         if mode==1:
#             selected=self.bestOf(arg[0])
#             for _ in range(self.populationSize):
#                 newPop.append(self.onePointCross(selected[random.randint(0, len(selected) - 1)]))
#         if mode==2:
#             selected = self.bestOf(arg[0])
#             for _ in range(self.populationSize):
#                 newPop.append(self.twoPointCross(selected[random.randint(0, len(selected) - 1)]))
#         return newPop
#
#
#     def bestOf(self,n):
#         return self.population[:n]
#
#     def onePointCross(self, parent:Chromosome):
#         result = parent.route
#         if np.random.rand() < self.crossChance:
#             i = np.random.randint(0, self.populationSize)
#
#             # zabezpieczenie przed wybraniem tego samego rodzica
#             while(parent == self.population[i]):
#                 i = np.random.randint(0, self.populationSize)
#
#             crossPoints = np.random.randint(0, len(parent.route))  # choose crossover points
#             result = parent.route[0:crossPoints] + [x for x in self.population[i].route if x not in  parent.route[0:crossPoints]]
#         return result
#
# def twoPointCross(self,parent:Chromosome):
#     result = parent.route
#     if np.random.rand() < self.crossChance:
#         i = np.random.randint(0, self.populationSize)
#
#         # zabezpieczenie przed wybraniem tego samego rodzica
#         while(parent == self.population[i]):
#             i = np.random.randint(0, self.populationSize)
#
#         crossPoint1 = np.random.randint(0, len(parent.route)) # choose crossover points
#         crossPoint2 =  np.random.randint(0, len(parent.route))
#         while(crossPoint1==crossPoint2):
#             crossPoint2 = np.random.randint(0, len(parent.route))
#         crossPoints=[crossPoint1,crossPoint2]
#         crossPoints.sort()
#         result = parent.route[0:crossPoints[0]] + parent.route[crossPoints[1]:]
#         missingNodes = [x for x in self.population[i].route if x not in result]
#
#         place=crossPoints[0]
#
#         for node in missingNodes:
#             result.insert(place,node)
#             place+=1
#     return result
#
#     def reinstateDepods(self, child):
#         cap = 0
#         route = [1]
#         for node in child:
#             if cap + self.data['nodes'][node]['demand'] < self.data['capacity']:
#                 route.append(node)
#                 cap += self.data['nodes'][node]['demand']
#             else:
#                 route.append(1)
#                 route.append(node)
#                 cap = self.data['nodes'][node]['demand']
#         route.append(1)
#         return route
#
#     def mutation(self, population):
#         for route in population:
#             for i in range(0,len(route)):
#                 if np.random.rand() < self.mutationChance:
#                     temp=route[i]
#                     swapPoint=np.random.randint(0,len(route))
#                     route[i]=route[swapPoint]
#                     route[swapPoint]=temp
#         return population
#
#     def mutationImproved(self,population):
#         for route in range(len(population)):
#             for i in range(len(population[route])):
#                 if np.random.rand() < self.mutationChance:
#                     bestScore=self.checkScore(population[route])
#                     currentPlacement=i
#                     tempNode=population[route].pop(i)
#                     for j in range(len(population[route])):
#                         population[route].insert(j,tempNode)
#                         currentScore=self.checkScore(population[route])
#                         if bestScore>currentScore:
#                             bestScore=currentScore
#                             currentPlacement=j
#                         population[route].pop(j)
#                     population[route].insert(currentPlacement,tempNode)
#         return population
#
#     def mutation2Try(self,population):
#         for route in range(len(population)):
#             for i in range(len(population[route])):
#                 if np.random.rand() < self.mutationChance:
#                     tempNode = population[route].pop(i)
#                     distances = [self.distance(self.data['nodes'][tempNode],self.data['nodes'][x]) for x in self.data['nodes'].keys()]
#                     best=distances[0]+distances[population[route][0]-1]
#                     bestPlacement=0
#                     for i in range(1,len(population[route])):
#                         currentScore=distances[population[route][i-1]-1]+distances[population[route][1]-1]
#                         if currentScore<best:
#                             best=currentScore
#                             bestPlacement=i
#                     population[route].insert(bestPlacement,tempNode)
#         return population
#     def returnBest(self) -> Chromosome:
#         return self.population[0]
#
# def plot(data,route):
#     route.pop(0)
#     x = [data['nodes'][1]['x']]
#     y = [data['nodes'][1]['y']]
#     for node in route:
#         if node==1:
#             x.append(data['nodes'][node]['x'])
#             y.append(data['nodes'][node]['y'])
#             plt.plot(x,y, marker='.')
#             x = [data['nodes'][node]['x']]
#             y = [data['nodes'][node]['y']]
#         else:
#             x.append(data['nodes'][node]['x'])
#             y.append(data['nodes'][node]['y'])
#     plt.show()
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
#
#
# if __name__=="__main__":
#     main()
# f"../cmtResult/resultsCMT{fileIndex}.xml"
# f"../Data/CMT/CMT{fileIndex}.vrp"
# print(f"resultsCMT{fileIndex}.xml   Running test for cOpt={cOpt} mOpt={mOpt}  pOpt={pOpt}  gOpt={gOpt}")

# class Line(object):
#     def __init__(self, n_moves, goal_point, start_point, obstacle_line):
#         self.n_moves = n_moves
#         self.goal_point = goal_point
#         self.start_point = start_point
#         self.obstacle_line = obstacle_line
#
#         plt.ion()
#
#     def plotting(self, lines_x, lines_y):
#         plt.cla()
#         plt.scatter(*self.goal_point, s=200, c='r')
#         plt.scatter(*self.start_point, s=100, c='b')
#         plt.plot(self.obstacle_line[:, 0], self.obstacle_line[:, 1], lw=3, c='k')
#         plt.plot(lines_x.T, lines_y.T, c='k')
#         plt.xlim((-5, 15))
#         plt.ylim((-5, 15))
#         plt.pause(0.01)
#
#
# ga = GA(DNA_size=DNA_SIZE, DNA_bound=DIRECTION_BOUND,
#         cross_rate=CROSS_RATE, mutation_rate=MUTATE_RATE, pop_size=POP_SIZE)
#
# env = Line(N_MOVES, GOAL_POINT, START_POINT, OBSTACLE_LINE)
#
# for generation in range(N_GENERATIONS):
#     lx, ly = ga.DNA2product(ga.pop, N_MOVES, START_POINT)
#     fitness = ga.get_fitness(lx, ly, GOAL_POINT, OBSTACLE_LINE)
#     ga.evolve(fitness)
#     print('Gen:', generation, '| best fit:', fitness.max())
#     env.plotting(lx, ly)
#
# plt.ioff()
# plt.show()

# def main():
#     #def __init__(self, file, crossChance, mutationChance, populationSize, genCount, function):
#     algorithm = GeneticAlgorithm("../Data/CMT/CMT1.vrp",0.01,0.02,50,100,Extract)
#     algorithm.run(1, 5)#metoda selekcji i n best
#     print(algorithm.returnBest())
#     plot(algorithm.data, algorithm.returnBest().route)
#
# if __name__ =="__main__":
#     main()

# def main():
#     algorithm=GeneticAlgorithm("christofides and elion/E-n22-k4.vrp",0.06,0.02,100,2,Extract)
#     algorithm.run(1,5)
#     print(algorithm.returnBest())
#
# if __name__=="__main__":
#     main()
