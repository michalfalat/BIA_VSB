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


def CalcPopulationDistance(population):
    distance = 0
    for i in range(len(population) - 1):
        distance += CalcCityDistance(population[i], population[i+1])
    distance += CalcCityDistance(population[len(population)-1], population[0])
    return distance


def Mutation(population, idx1, idx2):
    tmpCity = population[idx1]
    population[idx1] = population[idx2]
    population[idx2] = tmpCity
    return population


def Crossover(p1, p2):
    cityLength = len(p1.cities)
    for i in range(cityLength):
            if (i >= start and i <= end):
                for j in range(cityLength):
                    City currentCity = i2.cities.get(j)
                    boolean replace = true
                    for (int k=0; k < nI.cities.size(); k++) {
                        if (k >= i and k <= end) continue
                        if (currentCity.equals(nI.cities.get(k))) {
                            replace = false;
                            break; }
                    }

                    if (replace)
                        nI.cities.set(i, currentCity);
                        break;
                }

        return nI


def GeneratePopulationOfCities(popSize, cities):
    population = []
    cityUsed = True
    for i in range(popSize):
        currentPop = cities[:]
        np.random.shuffle(currentPop)
        population.append(currentPop)
    return population


def GenericAlghorithm():
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

    population = GeneratePopulationOfCities(50, cities)
    for i in range(len(population)):
        currentPop = ''
        currentPopDistance = CalcPopulationDistance(population[i])
        for j in range(len(population[i])):

            currentPop += (population[i][j].name + ", ")
        print(currentPop + " distance: " + str(currentPopDistance))


    tMax = 100
    # TODO two-point crossover
    for t in range(tMax):



GenericAlghorithm()