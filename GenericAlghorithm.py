import numpy as np
import random
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
    for i in range(len(population) - 1):
        # CalcCityDistance(population[i], population[i+1])
        distance += distanceMatrix[population[i].order][population[i+1].order]
    # CalcCityDistance(population[len(population)-1], population[0])
    distance += distanceMatrix[population[len(
        population)-1].order][population[0].order]
    return distance


def FindPopulationMinimum(populations, distanceMatrix):
    minimum = CalcPopulationDistance(populations[0], distanceMatrix)
    minPop = populations[0]
    for i in range(len(populations)):
        tmpDistance = CalcPopulationDistance(populations[i], distanceMatrix)
        if(tmpDistance < minimum):
            minimum = tmpDistance
            minPop = populations[i]
    return minPop


def Mutation(population, idx1, idx2):
    tmpCity = population[idx1]
    population[idx1] = population[idx2]
    population[idx2] = tmpCity
    return population


def Crossover(p1, p2):

    newPop = p1
    cityLength = len(p1)
    start = random.randint(0, cityLength-1)
    end = random.randint(0, cityLength-1)
    if start > end:
        start, end = end, start
    for i in range(cityLength):
        if (i >= start and i <= end):
            for j in range(cityLength):
                currentCity = p2[j]
                replace = True
                for k in range(cityLength):
                    if (k >= i and k <= end):
                        continue
                    if (currentCity == newPop[k]):
                        replace = False
                        break

                if replace == True:
                    newPop[i] = currentCity
                    break
    return newPop


def GeneratePopulationOfCities(popSize, cities):
    population = []
    for i in range(popSize):
        currentPop = cities[:]
        np.random.shuffle(currentPop)
        population.append(currentPop)
    return population


def GenericAlghorithm():
    popSize = 50
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

    population = GeneratePopulationOfCities(popSize, cities)
    # for i in range(popSize):
    #     currentPop = ''
    #     currentPopDistance = CalcPopulationDistance(
    #         population[i], distanceMatrix)
    #     for j in range(len(population[i])):

    #         currentPop += (population[i][j].name + ", ")
    #     print(currentPop + " distance: " + str(currentPopDistance))

    tMax = 100

    cityCount = len(cities)
    minPop = FindPopulationMinimum(population, distanceMatrix)
    minDistance = CalcPopulationDistance(minPop, distanceMatrix)
    globalMinDistance = minDistance
    globalMinPop = minPop
    for t in range(tMax):
        newPopulation = []
        for i in range(popSize):

            rndParentIdx = i
            while rndParentIdx == i:
                rndParentIdx = random.randint(0, popSize-1)
            parent = population[rndParentIdx]
            newIndividual = Crossover(parent, population[i])
            rndIdx1 = random.randint(0, cityCount-1)
            rndIdx2 = random.randint(0, cityCount-1)

            newIndividual = Mutation(newIndividual, rndIdx1, rndIdx2)
            if(CalcPopulationDistance(newIndividual, distanceMatrix) < CalcPopulationDistance(population[i], distanceMatrix)):
                newPopulation.append(newIndividual)
            else:
                newPopulation.append(population[i])
            minPop = FindPopulationMinimum(newPopulation, distanceMatrix)
            minDistance = CalcPopulationDistance(minPop, distanceMatrix)
            if(minDistance < globalMinDistance):
                globalMinDistance = minDistance
                globalMinPop = minPop
        # print(globalMinDistance)
        population = []
        population = newPopulation

    # print(globalMinDistance)
    currentPop = ""
    for j in range(len(globalMinPop)):
        currentPop += (globalMinPop[j].name + ", ")
    print(currentPop + " distance: " + str(globalMinDistance))

    # TODO two-point crossover
    # for t in range(tMax):


GenericAlghorithm()
