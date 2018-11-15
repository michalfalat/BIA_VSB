import numpy as np
import random
import operator
import math as m
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits import mplot3d
from random import randint


class City:
    def __init__(self, x, y, name, order):
        self.x = x
        self.y = y
        self.name = name
        self.order = order


class Population:
    def __init__(self, cities, distance=0):
        self.cities = cities
        self.distance = distance

    def append(self, city):
        self.cities.append(city)

    def calculateDistance(self, distanceMatrix):
        self.distance = CalcPopulationDistance(self, distanceMatrix)


def CalcCityDistance(cityA, cityB):
    diffX = abs(cityA.x - cityB.x)
    diffY = abs(cityA.y - cityB.y)
    distance = m.sqrt(m.pow(diffX, 2) + m.pow(diffY, 2))
    return distance


def CalcDistanceMatrix(cities):
    distanceMatrix = np.zeros((len(cities), len(cities)))
    for i in range(len(cities)):
        for j in range(len(cities)):
            if(i == j):
                distanceMatrix[i][j] = 0
            else:
                distanceMatrix[i][j] = CalcCityDistance(cities[i], cities[j])
    return distanceMatrix


def CalcPopulationDistance(population, distanceMatrix):
    distance = 0
    for i in range(len(population.cities) - 1):
        # CalcCityDistance(population[i], population[i+1])
        distance += distanceMatrix[population.cities[i]
                                   .order][population.cities[i+1].order]
    # CalcCityDistance(population[len(population)-1], population[0])
    distance += distanceMatrix[population.cities[len(
        population.cities)-1].order][population.cities[0].order]
    return distance


def FindPopulationMinimum(populations, distanceMatrix):
    minimum = populations[0].distance
    minimumLocalPop = populations[0]
    for i in range(len(populations)):
        if(populations[i].distance < minimum):
            minimum = populations[i].distance
            minimumLocalPop = populations[i]
    return minimumLocalPop


def Mutation(population, idx1, idx2):
    tmpCity = population.cities[idx1]
    population.cities[idx1] = population.cities[idx2]
    population.cities[idx2] = tmpCity
    return population


def Crossover(chromosome1, chromosome2):
    end = randint(0, len(chromosome1.cities))
    start = randint(0, end)
    section = chromosome1.cities[start:end]
    offspring_genes = list(
        gene if gene not in section else None for gene in chromosome2.cities)
    g = (x for x in section)
    for i, x in enumerate(offspring_genes):
        if x is None:
            offspring_genes[i] = next(g)
    offspring = Population(offspring_genes)

    return offspring


def GeneratePopulationOfCities(popSize, cities, distanceMatrix):
    population = []
    for i in range(popSize):
        currentPop = cities[:]
        np.random.shuffle(currentPop)
        p = Population(currentPop)
        p.calculateDistance(distanceMatrix)
        population.append(p)
    return population


def ShowPlot(cities, population):
    plt.cla()
    x = [o.x for o in cities]
    y = [o.y for o in cities]
    plt.scatter(x, y, zorder=2)
    for i in range(len(cities)):
        plt.annotate(cities[i].name, (x[i], y[i]))

    for i in range(len(population.cities)-1):
        linesX = [population.cities[i].x, population.cities[i+1].x]
        linesY = [population.cities[i].y, population.cities[i+1].y]
        plt.plot(linesX, linesY, zorder=1)
    linesXLast = [population.cities[len(cities)-1].x, population.cities[0].x]
    linesYLast = [population.cities[len(cities)-1].y, population.cities[0].y]
    plt.plot(linesXLast, linesYLast, zorder=1)
    plt.pause(0.1)


def GenericAlghorithm():
    popSize = 100
    cities = []
    cities.append(City(60, 200, 'A', 0))
    cities.append(City(80, 200, 'B', 1))
    cities.append(City(80, 180, 'C', 2))
    cities.append(City(140, 180, 'D', 3))
    cities.append(City(20, 160, 'E', 4))
    cities.append(City(100, 160, 'F', 5))
    cities.append(City(200, 160, 'G', 6))
    cities.append(City(140, 140, 'H', 7))
    cities.append(City(40, 120, 'I', 8))
    cities.append(City(100, 120, 'J', 9))
    cities.append(City(180, 100, 'K', 10))
    cities.append(City(60, 80, 'L', 11))
    cities.append(City(120, 80, 'M', 12))
    cities.append(City(180, 60, 'N', 13))
    cities.append(City(20, 40, 'O', 14))
    cities.append(City(100, 40, 'P', 15))
    cities.append(City(200, 40, 'Q', 16))
    cities.append(City(20, 20, 'R', 17))
    cities.append(City(60, 20, 'S', 18))
    cities.append(City(160, 20, 'T', 19))

    distanceMatrix = CalcDistanceMatrix(cities)

    population = GeneratePopulationOfCities(popSize, cities, distanceMatrix)
    # for i in range(popSize):
    #     currentPop = ''
    #     currentPopDistance = CalcPopulationDistance(
    #         population[i], distanceMatrix)
    #     for j in range(len(population[i])):

    #         currentPop += (population[i][j].name + ", ")
    #     print(currentPop + " distance: " + str(currentPopDistance))

    tMax = 500

    C_param = 0.9
    M_param = 0.08
    cityCount = len(cities)
    globalMinPop = FindPopulationMinimum(population, distanceMatrix)
    #minDistance = CalcPopulationDistance(minPop, distanceMatrix)
    #globalMinDistance = minDistance
    #globalMinPop = minPop
    for t in range(tMax):
        newPopulation = []
        for i in range(popSize):
            rndCrossover = random.uniform(0, 1)
            rndMutation = random.uniform(0, 1)

            if rndCrossover < C_param:
                rndParentIdx = i
                while rndParentIdx == i:
                    rndParentIdx = random.randint(0, popSize/2)
                parent = population[rndParentIdx]
                newIndividual = Crossover(parent, population[i])
            else:
                newIndividual = population[i]

            if rndMutation < M_param:
                rndIdx1 = random.randint(0, cityCount-1)
                rndIdx2 = random.randint(0, cityCount-1)
                newIndividual = Mutation(newIndividual, rndIdx1, rndIdx2)
            newIndividual.calculateDistance(distanceMatrix)
            population[i].calculateDistance(distanceMatrix)

            if(newIndividual.distance < population[i].distance):
                #pop = Population(newIndividual.cities)
                # pop.calculateDistance(distanceMatrix)
                newPopulation.append(newIndividual)
            else:
                #pop = Population(population[i].cities)
                # pop.calculateDistance(distanceMatrix)
                newPopulation.append(population[i])
        minPop = FindPopulationMinimum(newPopulation, distanceMatrix)
        minPop.calculateDistance(distanceMatrix)
        #minDistance = CalcPopulationDistance(minPop, distanceMatrix)
        if(minPop.distance < globalMinPop.distance):
            globalMinPop = Population(minPop.cities)
            globalMinPop.calculateDistance(distanceMatrix)
            print("new minimum:" + str(minPop.distance))
            ShowPlot(cities, globalMinPop)
        # print(globalMinDistance)
        population = []
        # newPopulation.sort(key = operator.itemgetter(1))
        population = sorted(
            newPopulation, key=lambda x: x.distance, reverse=False)

    # print(globalMinDistance)

    currentPop = ""
    for j in range(len(globalMinPop.cities)):
        currentPop += (globalMinPop.cities[j].name + ", ")
    print(currentPop + " distance: " + str(globalMinPop.distance))
    plt.show()


GenericAlghorithm()
