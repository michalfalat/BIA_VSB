import math
import numpy as np


class MOO:
    def __init__(self, x):
        self.x = x
        self.f1 = func1(x)
        self.f2 = func2(x)


def func1(x):
    return  - math.pow(x, 2)

def func2(x):
    return  - math.pow((x-2), 2)

def generatePopulation(popSize):
    population = []
    for i in range(popSize):
        rand = np.random.randint(-10, 10, 1)[0]
        population.append(MOO(rand))
    return population




def main():
    OPT_TYPE = 'MAX'
    popSize = 10
    population = generatePopulation(popSize)

    Np = []
    Sp = []
    for i in range(popSize):
        NpCounter = 0        
        for j in range(popSize):
            if OPT_TYPE == 'MAX':
                if population[i].f1 < population[j].f1 and population[i].f2 < population[j].f2:
                    NpCounter += 1
        Np.append(NpCounter)

            #else:
    
    print('a')

main()