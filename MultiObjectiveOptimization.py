import math
import numpy as np


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
        # rand = np.random.randint(-10, 10, 1)[0]
        # population.append(MOO(rand))
        population.append(MOO(populationTempl[i]))
    return population


def main():
    OPT_TYPE = 'MAX'
    popSize = 6
    population = generatePopulation()

    # Np = {4, 2, 0, 0, 3, 0}
    # Sp = {{ }, {0}, {0, 1, 4}, {0, 4}, { }, {0, 1, 4}}

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

    print('a')


main()
