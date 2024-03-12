import numpy as np
from Data.Extractor import Extract


def distance(node1, node2):
    return np.sqrt(np.sum((node1 - node2) ** 2))


class AntColony:
    def __init__(self, file, nAnt, nIter, a, b, evapRate, Q, Extract):
        self.data = Extract(file)
        self.nodeCount = len(self.data['nodes'].keys())
        self.pherMatrix = np.ones((self.nodeCount, self.nodeCount))
        self.bestRoute = None
        self.bestScore = np.inf
        self.a = a
        self.b = b
        self.evapRate = evapRate
        self.Q = Q
        self.nAnt = nAnt
        self.nIter = nIter
        self.nodes = self.collectNodes()

    def collectNodes(self):
        ar = np.empty(shape=[0, 2])
        for elem in self.data['nodes'].keys():
            ar = np.append(ar, [[self.data['nodes'][elem]['x'], self.data['nodes'][elem]['y']]], axis=0)
        return ar

    def run(self):
        for iter in range(self.nIter):
            routes = []
            routeScores = []

            for ant in range(self.nAnt):
                visited = [False] * self.nodeCount
                currentNode = 0
                visited[currentNode] = True
                route = [currentNode]
                cap = 0
                score = 0
                while False in visited:
                    unvisited = np.where(np.logical_not(visited))[0]
                    prob = np.zeros(len(unvisited))
                    for i, unvisitedPoint in enumerate(unvisited):
                        prob[i] = (self.pherMatrix[currentNode, unvisitedPoint] ** self.a /
                                   (distance(self.nodes[currentNode], self.nodes[unvisitedPoint]) ** self.b))
                    prob /= np.sum(prob)
                    nextNode = np.random.choice(unvisited, p=prob)
                    if (cap + self.data['nodes'][nextNode + 1]['demand']) > self.data['capacity']:
                        route.append(0)
                        cap = self.data['nodes'][nextNode + 1]['demand']
                        score += distance(self.nodes[currentNode], self.nodes[0])
                        score += distance(self.nodes[0], self.nodes[nextNode])
                    else:
                        cap += self.data['nodes'][nextNode + 1]['demand']
                        score += distance(self.nodes[currentNode], self.nodes[nextNode])
                    route.append(nextNode)
                    visited[nextNode] = True
                    currentNode = nextNode
                score += distance(self.nodes[0], self.nodes[route[-1]])
                route.append(0)
                routes.append(route)
                routeScores.append(score)
                if score < self.bestScore:
                    self.bestScore = score
                    self.bestRoute = route
            self.pherMatrix *= self.evapRate
            for route, score in zip(routes, routeScores):
                for i in range(self.nodeCount - 1):
                    self.pherMatrix[route[i], route[i + 1]] += self.Q / score

# f"../cmtResultAnt/resultsCMT{fileIndex}.xml"
# f"../Data/CMT/CMT{fileIndex}.vrp"
# print(f"resultsCMT{fileIndex}.xml   Running test for cOpt={cOpt} mOpt={mOpt}  pOpt={pOpt}  gOpt={gOpt}")

# def main():
#     antsC = (25, 50)
#     iterCount = (20, 40)
#     # files = (1, 2, 3, 4, 5, 11, 12)
#     files = (4, 5, 11, 12)
#     for fileIndex in files:
#         file = open(f"../cmtResultAnt/resultsCMT{fileIndex}.xml", "w")
#         file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
#         file.write('<results>\n')
#         file.close()
#         for nAnt in antsC:
#             for nIter in iterCount:
#                 print(f"Running test for nAnt={nAnt} nInter={nIter}")
#                 file = open(f"../cmtResultAnt/resultsCMT{fileIndex}.xml", "a+")
#                 file.write(f'<test nAnt="{nAnt}" nInter="{nIter}">\n')
#                 for i in range(5):
#                     print(f"resultsCMT{fileIndex}.xml  Running test {i + 1}")
#                     # __init__(self,file,nAnt,nIter,a,b,evapRate,Q,Extract)
#                     antGen = AntColony(f"../Data/CMT/CMT{fileIndex}.vrp", nAnt, nIter, 1, 1, 0.5, 1, Extract)
#                     antGen.run()
#                     file.write(f'<route id="{i}" score="{antGen.bestScore}">\n')
#                     file.write(str(list((np.array(antGen.bestRoute)+1))))
#                     file.write("\n")
#                     file.write('</route>\n')
#                 file.write('</test>\n')
#                 file.close()
#         file = open(f"../cmtResultAnt/resultsCMT{fileIndex}.xml", "a+")
#         file.write('</results>\n')
#         file.close()
#
#
# if __name__ == "__main__":
#     main()

# import random as rn
# import numpy as np
# from numpy.random import choice as np_choice
#
# class AntColony(object):
#
#     def __init__(self, distances, n_ants, n_best, n_iterations, decay, alpha=1, beta=1):
#         """
#         Args:
#             distances (2D numpy.array): Square matrix of distances. Diagonal is assumed to be np.inf.
#             n_ants (int): Number of ants running per iteration
#             n_best (int): Number of best ants who deposit pheromone
#             n_iteration (int): Number of iterations
#             decay (float): Rate it which pheromone decays. The pheromone value is multiplied by decay, so 0.95 will lead to decay, 0.5 to much faster decay.
#             alpha (int or float): exponent on pheromone, higher alpha gives pheromone more weight. Default=1
#             beta (int or float): exponent on distance, higher beta give distance more weight. Default=1
#
#         Example:
#             ant_colony = AntColony(german_distances, 100, 20, 2000, 0.95, alpha=1, beta=2)
#         """
#         self.distances = distances
#         self.pheromone = np.ones(self.distances.shape) / len(distances)
#         self.all_inds = range(len(distances))
#         self.n_ants = n_ants
#         self.n_best = n_best
#         self.n_iterations = n_iterations
#         self.decay = decay
#         self.alpha = alpha
#         self.beta = beta
#
#     def run(self):
#         shortest_path = None
#         all_time_shortest_path = ("placeholder", np.inf)
#         for i in range(self.n_iterations):
#             all_paths = self.gen_all_paths()
#             self.spread_pheromone(all_paths, self.n_best, shortest_path=shortest_path)
#             shortest_path = min(all_paths, key=lambda x: x[1])
#             print (shortest_path)
#             if shortest_path[1] < all_time_shortest_path[1]:
#                 all_time_shortest_path = shortest_path
#             self.pheromone = self.pheromone * self.decay
#         return all_time_shortest_path
#
#     def spread_pheromone(self, all_paths, n_best, shortest_path):
#         sorted_paths = sorted(all_paths, key=lambda x: x[1])
#         for path, dist in sorted_paths[:n_best]:
#             for move in path:
#                 self.pheromone[move] += 1.0 / self.distances[move]
#
#     def gen_path_dist(self, path):
#         total_dist = 0
#         for e in path:
#             total_dist += self.distances[e]
#         return total_dist
#
#     def gen_all_paths(self):
#         all_paths = []
#         for i in range(self.n_ants):
#             path = self.gen_path(0)
#             all_paths.append((path, self.gen_path_dist(path)))
#         return all_paths
#
#     def gen_path(self, start):
#         path = []
#         visited = set()
#         visited.add(start)
#         prev = start
#         for i in range(len(self.distances) - 1):
#             move = self.pick_move(self.pheromone[prev], self.distances[prev], visited)
#             path.append((prev, move))
#             prev = move
#             visited.add(move)
#         path.append((prev, start)) # going back to where we started
#         return path
#
#     def pick_move(self, pheromone, dist, visited):
#         pheromone = np.copy(pheromone)
#         pheromone[list(visited)] = 0
#
#         row = pheromone ** self.alpha * (( 1.0 / dist) ** self.beta)
#
#         norm_row = row / row.sum()
#         move = np_choice(self.all_inds, 1, p=norm_row)[0]
#         return move
