import random
import operator
import math


import matplotlib.pyplot as plt

class City:
    def __init__(self, x, y, name, order):
        self.x = x
        self.y = y
        self.name = name
        self.order = order

class Graph(object):
    def __init__(self, cost_matrix: list, rank: int):
        self.matrix = cost_matrix
        self.rank = rank
        # noinspection PyUnusedLocal
        self.pheromone = [[1 / (rank * rank) for j in range(rank)] for i in range(rank)]


class ACO(object):
    def __init__(self, ant_count: int, generations: int, alpha: float, beta: float, rho: float, q: int,
                 strategy: int):
        self.Q = q
        self.rho = rho
        self.beta = beta
        self.alpha = alpha
        self.ant_count = ant_count
        self.generations = generations
        self.update_strategy = strategy

    def _update_pheromone(self, graph: Graph, ants: list):
        for i, row in enumerate(graph.pheromone):
            for j, col in enumerate(row):
                graph.pheromone[i][j] *= self.rho
                for ant in ants:
                    graph.pheromone[i][j] += ant.pheromone_delta[i][j]

    # noinspection PyProtectedMember
    def solve(self, graph: Graph, cities):
        best_cost = float('inf')
        best_solution = []
        for gen in range(self.generations):
            # noinspection PyUnusedLocal
            ants = [_Ant(self, graph) for i in range(self.ant_count)]
            for ant in ants:
                for i in range(graph.rank - 1):
                    ant._select_next()
                ant.total_cost += graph.matrix[ant.tabu[-1]][ant.tabu[0]]
                if ant.total_cost < best_cost:
                    best_cost = ant.total_cost
                    best_solution = [] + ant.tabu
                    print("new minimum:" + str(best_cost))
                show_plot(cities, ant.tabu, best_solution)
                # update pheromone
                ant._update_pheromone_delta()
            self._update_pheromone(graph, ants)
            # print('generation #{}, best cost: {}, path: {}'.format(gen, best_cost, best_solution))
        return best_solution, best_cost


class _Ant(object):
    def __init__(self, aco: ACO, graph: Graph):
        self.colony = aco
        self.graph = graph
        self.total_cost = 0.0
        self.tabu = []  # tabu list
        self.pheromone_delta = []  # the local increase of pheromone
        self.allowed = [i for i in range(graph.rank)]  # nodes which are allowed for the next selection
        self.eta = [[0 if i == j else 1 / graph.matrix[i][j] for j in range(graph.rank)] for i in
                    range(graph.rank)]  # heuristic information
        start = random.randint(0, graph.rank - 1)  # start from any node
        self.tabu.append(start)
        self.current = start
        self.allowed.remove(start)

    def _select_next(self):
        denominator = 0
        for i in self.allowed:
            denominator += self.graph.pheromone[self.current][i] ** self.colony.alpha * self.eta[self.current][
                                                                                            i] ** self.colony.beta
        # noinspection PyUnusedLocal
        probabilities = [0 for i in range(self.graph.rank)]  # probabilities for moving to a node in the next step
        for i in range(self.graph.rank):
            try:
                self.allowed.index(i)  # test if allowed list contains i
                probabilities[i] = self.graph.pheromone[self.current][i] ** self.colony.alpha * \
                    self.eta[self.current][i] ** self.colony.beta / denominator
            except ValueError:
                pass  # do nothing
        # select next node by probability roulette
        selected = 0
        rand = random.random()
        for i, probability in enumerate(probabilities):
            rand -= probability
            if rand <= 0:
                selected = i
                break
        self.allowed.remove(selected)
        self.tabu.append(selected)
        self.total_cost += self.graph.matrix[self.current][selected]
        self.current = selected

    # noinspection PyUnusedLocal
    def _update_pheromone_delta(self):
        self.pheromone_delta = [[0 for j in range(self.graph.rank)] for i in range(self.graph.rank)]
        for _ in range(1, len(self.tabu)):
            i = self.tabu[_ - 1]
            j = self.tabu[_]
            if self.colony.update_strategy == 1:  # ant-quality system
                self.pheromone_delta[i][j] = self.colony.Q
            elif self.colony.update_strategy == 2:  # ant-density system
                # noinspection PyTypeChecker
                self.pheromone_delta[i][j] = self.colony.Q / self.graph.matrix[i][j]
            else:  # ant-cycle system
                self.pheromone_delta[i][j] = self.colony.Q / self.total_cost


def show_plot(points, path: list, best):
    # plt.cla()
    # x = [o.x for o in cities]
    # y = [o.y for o in cities]
    # plt.scatter(x, y, zorder=2)
    # for i in range(len(cities)):
    #     plt.annotate(cities[i].name, (x[i], y[i]))

    # for i in range(len(population.cities)-1):
    #     linesX = [population.cities[i].x, population.cities[i+1].x]
    #     linesY = [population.cities[i].y, population.cities[i+1].y]
    #     plt.plot(linesX, linesY, zorder=1)
    # linesXLast = [population.cities[len(cities)-1].x, population.cities[0].x]
    # linesYLast = [population.cities[len(cities)-1].y, population.cities[0].y]
    # plt.plot(linesXLast, linesYLast, zorder=1)
    # plt.pause(0.1)
    plt.cla()
    x = []
    y = []
    for point in points:
        x.append(point['x'])
        y.append(point['y'])
    # noinspection PyUnusedLocal
    y = list(map(operator.sub, [max(y) for i in range(len(points))], y))
    #plt.plot(x, y, zorder=1)
    plt.scatter(x, y, zorder=2)

    for _ in range(1, len(path)):
        i = path[_ - 1]
        j = path[_]
        # noinspection PyUnresolvedReferences
        plt.arrow(x[i], y[i], x[j] - x[i], y[j] - y[i],color='gray', length_includes_head=True)
    
    i = path[len(path)-1]
    j = path[0]
        # noinspection PyUnresolvedReferences
    plt.arrow(x[i], y[i], x[j] - x[i], y[j] - y[i],color='gray', length_includes_head=True)
    
    #plt.arrow(x[len(path)-1], y[len(path)-1], x[0] - x[len(path)-1], y[0] - y[len(path)-1],color='gray', length_includes_head=True)

    #  for i in range(len(population.cities)-1):
    #     linesX = [population.cities[i].x, population.cities[i+1].x]
    #     linesY = [population.cities[i].y, population.cities[i+1].y]
    #     plt.plot(linesX, linesY, zorder=1)
    # linesXLast = [population.cities[len(cities)-1].x, population.cities[0].x]
    # linesYLast = [population.cities[len(cities)-1].y, population.cities[0].y]
    # plt.plot(linesXLast, linesYLast, zorder=1)

    for _ in range(1, len(best)):
        i = best[_ - 1]
        j = best[_]
        # noinspection PyUnresolvedReferences
        plt.arrow(x[i], y[i], x[j] - x[i], y[j] - y[i], color='r', length_includes_head=True)

    i = best[len(path)-1]
    j = best[0]
        # noinspection PyUnresolvedReferences
    plt.arrow(x[i], y[i], x[j] - x[i], y[j] - y[i],color='r', length_includes_head=True)


    # noinspection PyTypeChecker
    plt.xlim(0, max(x) * 1.1)
    # noinspection PyTypeChecker
    plt.ylim(0, max(y) * 1.1)
    plt.pause(0.01)

def calcCityDistance(cityA, cityB):
    diffX = abs(cityA['x'] - cityB['x'])
    diffY = abs(cityA['y'] - cityB['y'])
    distance = math.sqrt(math.pow(diffX, 2) + math.pow(diffY, 2))
    return distance

def generateCities():
    cities = []
    cities.append(dict(index = 0, x = int(60), y= int(200),name ='A' ))
    cities.append(dict(index = 1, x = int(80), y= int(200),name ='B' ))
    cities.append(dict(index = 2, x = int(80), y= int(180),name ='C' ))
    cities.append(dict(index = 3, x = int(140),y= int(180),name ='D' ))
    cities.append(dict(index = 4, x = int(20), y= int(160),name ='E' ))
    cities.append(dict(index = 5, x = int(100),y= int(160),name ='F' ))
    cities.append(dict(index = 6, x = int(200),y= int(160),name ='G' ))
    cities.append(dict(index = 7, x = int(140),y= int(140),name ='H' ))
    cities.append(dict(index = 8, x = int(40), y= int(120),name ='I' ))
    cities.append(dict(index = 9, x = int(100),y= int(120),name ='J' ))
    cities.append(dict(index = 10, x = int(180),y= int(100),name = 'K'))
    cities.append(dict(index = 11, x = int(60), y= int(80), name = 'L'))
    cities.append(dict(index = 12, x = int(120),y= int(80), name = 'M'))
    cities.append(dict(index = 13, x = int(180),y= int(60), name = 'N'))
    cities.append(dict(index = 14, x = int(20), y= int(40), name = 'O'))
    cities.append(dict(index = 15, x = int(100),y= int(40), name = 'P'))
    cities.append(dict(index = 16, x = int(200),y= int(40), name = 'Q'))
    cities.append(dict(index = 17, x = int(20), y= int(20), name = 'R'))
    cities.append(dict(index = 18, x = int(60), y= int(20), name = 'S'))
    cities.append(dict(index = 19, x = int(160),y= int(20), name = 'T'))
    return cities

def aco():
    cities = generateCities()
    points = []
    
    cost_matrix = []
    rank = len(cities)
    for i in range(rank):
        row = []
        for j in range(rank):
            row.append(calcCityDistance(cities[i], cities[j]))
        cost_matrix.append(row)
    #ant_count: int, generations: int, alpha: float, beta: float, rho: float, q: int, strategy: int):
    aco = ACO(10, 100, 1.0, 10.0, 0.5, 10, 2)
    graph = Graph(cost_matrix, rank)
    path, cost = aco.solve(graph, cities)
    print('cost: {}, path: {}'.format(cost, path))
    #show_plot(cities, path)



aco()
