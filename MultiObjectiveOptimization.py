import math
import numpy as np
import random


class MOO:
    def __init__(self, x):
        self.x = x
        self.f1 = func1(x)
        self.f2 = func2(x)


def func1(x):
    return - math.pow(x, 2)


def func2(x):
    return - math.pow((x-2), 2)


def generatePopulation(popSize=6):
    populationTempl = [-2, -1, 0, 2, 4, 1]
    population = []
    for i in range(popSize):
        rand = np.random.randint(-55, 55, 1)[0]
        # population.append(MOO(rand))
        population.append(MOO(populationTempl[i]))
    return population


def generateNewPopulation(old_pop, delta):

    new_pop = []
    for i in range(len(old_pop)):
        parents = random.sample(old_pop, 2)
        x = (parents[0].f1 + parents[1].f1) / 2
        y = (parents[0].f2 + parents[1].f2) / 2

        x += delta
        y += delta
        newObj = MOO(0)
        newObj.f1 = x
        newObj.f2 = y
        new_pop.append(newObj)

    return new_pop


def reducePopulation(pop, size):
    sorted_pop = []
    sorted_pop.append(pop[0])
    for i in range(1, len(pop)):
        actual = pop[i]

        for j in range(len(sorted_pop)):
            sort = sorted_pop[j]

            if actual.f1 >= sort.f1 and actual.f2 >= sort.f2:
                sorted_pop.insert(j, actual)
                break
            elif j == len(sorted_pop) - 1:
                sorted_pop.insert(j+1, actual)

    return sorted_pop[:size]


# Fast nondominated sorting
def rank(population, popSize, OPT_TYPE):
    Np = []
    Sp = []
    Q1 = []
    for i in range(popSize):
        NpCounter = 0
        SpArray = []
        for j in range(popSize):
            if OPT_TYPE == 'MAX':
                if i != j and population[i].f1 <= population[j].f1 and population[i].f2 <= population[j].f2:
                    NpCounter += 1
                if i != j and population[i].f1 >= population[j].f1 and population[i].f2 >= population[j].f2:
                    SpArray.append(j)
            if OPT_TYPE == 'MIN':
                if i != j and population[i].f1 >= population[j].f1 and population[i].f2 >= population[j].f2:
                    NpCounter += 1
                if i != j and population[i].f1 <= population[j].f1 and population[i].f2 <= population[j].f2:
                    SpArray.append(j)

        Np.append(NpCounter)
        Sp.append(SpArray)
        if NpCounter == 0:
            Q1.append(len(Np)-1)

    Q = []
    i = 1
    while Q1 != []:
        Q.append(Q1)
        F = []
        for q in range(len(Q1)):
            for s in range(len(Sp)):
                Np[s] = Np[s] - 1
                if(Np[s] == 0):
                    q_rank = i+1
                    F.append(s)
        i = i+1
        Q1 = F
    return Q


def main():
    OPT_TYPE = 'MAX'
    popSize = 6
    population = generatePopulation()

    # Np = {4, 2, 0, 0, 3, 0}
    # Sp = {{ }, {0}, {0, 1, 4}, {0, 4}, { }, {0, 1, 4}}
    Q = rank(population, popSize, OPT_TYPE)
    print(Q)

    child_pop = generateNewPopulation(population, 0.002)

    joined_pop = population.copy()
    joined_pop.extend(child_pop)
    final_pop = reducePopulation(joined_pop, popSize)

    Q = rank(final_pop, popSize, OPT_TYPE)

    print(Q)


main()
