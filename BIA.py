import numpy as np
import random
import math as m
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits import mplot3d
from random import randint


class IndividualPoint:
    def __init__(self, coordinates, z=0):
        self.coordinates = coordinates
        self.z = z

    def setZ(self, z):
        self.z = z

    def __str__(self):
        return ("IndividualPoint on:",self.coordinates,"With z:",self.z)

class Particle:
    def __init__(self, coordinates, z=0, velocity=[0,0]):
        self.coordinates = coordinates
        self.z = z
        self.velocity = velocity

    def setZ(self, z):
        self.z = z

    def setVelocity(self, velocity):
        self.velocity = velocity

    def __str__(self):
        return ("Particle on:",self.coordinates,"With z:",self.z)


def SphereFunction(coordinates):
    result = 0.0
    for i in range(0,len(coordinates)):
        result += m.pow(coordinates[i],2)

    return result

def RastriginFunction(coordinates):
    dim = len(coordinates)
    result = 0.0
    result += 10*dim
    for i in range(0,dim):
        temp = pow(coordinates[i],2) - 10 * m.cos(2*m.pi * coordinates[i])
        result += temp

    return result

def RosenbrockFunction(coordinates):
    dim = len(coordinates)
    result = 0.0
    for i in range(0,dim-1):
        temp = 100* pow((coordinates[i+1]-pow(coordinates[i],2)),2) + pow((1 -coordinates[i]),2)
        result += temp

    return result

def AckleyFunction(coordinates):
    dim = len(coordinates)
    a = 20
    d = 0.2
    c = 2 * m.pi
    temp1 = 0.0
    temp2 = 0.0
    for i in range(0,dim):
        temp1 += m.pow(coordinates[i],2)
        temp2 += m.cos(c*coordinates[i])

    temp11 = -a * m.exp(-0.2 * m.pow((0.5*temp1), 1/2))
    temp21 = m.exp(0.5 * temp2)
    result = temp11 - temp21 + a + m.exp(1)
    return result

def SchwefelFunction(coordinates):
    dim = len(coordinates)
    result = 0.0
    for i in range(0,dim):
        result += coordinates[i] * m.sin(m.sqrt(abs(coordinates[i])))

    result = 418.9829 * dim - result
    return result


def FindGlobalMininum(individualPoints):
    individualPoint = individualPoints[0]
    for i in range(len(individualPoints)):
        if(individualPoints[i].z < individualPoint.z):
            individualPoint = individualPoints[i]
    #print(individualPoint.z)
    return individualPoint

def FindGlobalMininumIndex(individualPoints):
    index = 0
    individualPoint = individualPoints[0]
    for i in range(len(individualPoints)):
        if(individualPoints[i].z < individualPoint.z):
            index = i
    #print(individualPoint.z)
    return index

def GenerateRandomIndividualPoint(defMin, defMax):
    ranCoordinates = (randint(defMin, defMax),randint(defMin, defMax))
    individualPoint = IndividualPoint(ranCoordinates)
    return individualPoint

def GenerateIndividualPointsInNeighbourhood(step, mainIndividualPoint):
    individualPoints = []
    coordinates =  np.random.multivariate_normal(mainIndividualPoint.coordinates,[[1,0],[0 ,100]],10)
    for i in range(len(coordinates)):
        individualPoints.append(IndividualPoint((coordinates[i][0], coordinates[i][1]))) 
    return individualPoints

def GenerateRandomIndividualPointForSA(mainIndividualPoint):
    coordinates =  np.random.multivariate_normal(mainIndividualPoint.coordinates,[[1,1],[1 ,1]],1)
    individualPoint = IndividualPoint((coordinates[0][0], coordinates[0][1]))
    return individualPoint

def GenerateRandomForSOMA(min,max):
    x = np.random.uniform(min,max,20)
    y = np.random.uniform(min,max,20)
    field = []
    for i in range(len(x)):
        field.append(IndividualPoint((x[i],y[i]),0))
    return field

def GenerateParticles(min,max,  initVelocity ,count = 20,):
    x = np.random.uniform(min,max,count)
    y = np.random.uniform(min,max,count)
    particles = []
    for i in range(len(x)):
        particles.append(Particle([x[i],y[i]],0,initVelocity))
    return particles

def GetPRT(dim,ptr_value):
    x = np.random.uniform(0,1,dim)
    for i in range(len(x)):
        if (x[i] < ptr_value):
            x[i] = 1
        else:
            x[i] = 0
    
    return x

def GetFunction(data, func):
    if func =='sphere':
        return SphereFunction(data)
    elif func == 'rastrigin':
        return RastriginFunction(data)
    elif func == 'rosenbrock':
        return RosenbrockFunction(data)
    elif func == 'ackley':
        return AckleyFunction(data)
    elif func == 'schwefel':
        return SchwefelFunction(data)
        

def SomaAlgorithm(func):
    pathLength = 3
    step = 0.11
    ptrValue = 0.1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(def_min, def_max, stepDrawing)
    X, Y = np.meshgrid(x, y)
    zs = np.array([GetFunction((x,y), func) for x,y in zip(np.ravel(X), np.ravel(Y))])  
    Z = zs.reshape(X.shape)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    field = GenerateRandomForSOMA(def_min, def_max)

    for i in range(len(field)):  
        field[i].setZ(GetFunction(field[i].coordinates, func))

    for u in range(len(field)):
        ax.scatter(field[u].coordinates[0],field[u].coordinates[1],field[u].z-0.2,color="black",s=20)
        
    
    leaderIndex = FindGlobalMininumIndex(field) 
    leader = field[leaderIndex]    

    ax.scatter(leader.coordinates[0],leader.coordinates[1],leader.z-0.2,color="r",s=20)

    for s in range(100):
        for k in range(len(field)):
            if (leader == field[k]):
                continue
            
            jumps = []
            for i in range(0,int(pathLength/step),1):
                dim = len(leader.coordinates)
                PRT = GetPRT(dim,ptrValue)
                newCoords = []
                for d in range(0,dim):
                    crd = field[k].coordinates[d] + (leader.coordinates[d] - field[k].coordinates[d]) * (i* step) * PRT[d]
                    newCoords.append(crd)
                w = IndividualPoint(newCoords,0)
                w.z = GetFunction(w.coordinates, func)
                jumps.append(w)
            
            _newLeaderIndex = FindGlobalMininumIndex(jumps)
            _new_pos = jumps[_newLeaderIndex]
            field[k] = _new_pos
        
        ax.clear()
        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        x = y = np.arange(def_min, def_max, stepDrawing)
        X, Y = np.meshgrid(x, y)
        zs = np.array([GetFunction((x,y), func) for x,y in zip(np.ravel(X), np.ravel(Y))])  
        Z = zs.reshape(X.shape)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

        for h in range(len(field)):
            ax.scatter( field[h].coordinates[0],field[h].coordinates[1],field[h].z-0.2,color="black",s=20)

        leaderIndex = FindGlobalMininumIndex(field) 
        leader = field[leaderIndex]

        ax.scatter(leader.coordinates[0],leader.coordinates[1],leader.z-0.2,color="r",s=20)
        plt.pause(0.2)
    plt.show()

def GetVelocity(c1,c2, weight,  vi, xi, pBest, gBest):
     return  weight * vi + (c1 * random.uniform(0,1)) * (pBest - xi) + (c2 * random.uniform(0,1)) * (gBest - xi)

def ParticleSwarm(func, c1, c2):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(def_min, def_max, stepDrawing)
    X, Y = np.meshgrid(x, y)
    zs = np.array([GetFunction((x,y), func) for x,y in zip(np.ravel(X), np.ravel(Y))])  
    Z = zs.reshape(X.shape)
    #ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    initialVelocity = (abs(def_min) + abs(def_max)) /20
    initialVelVector = [initialVelocity,initialVelocity]
    iterations = 100
    weightStart = 0.9
    weightEnd = 0.4

    particles = GenerateParticles(def_min, def_max,initialVelVector,20)
    for particle in particles:
        particle.setZ(GetFunction(particle.coordinates, func))
    
    gBest = FindGlobalMininum(particles)
    pBest = particles[0]        

    for i in range(iterations):
        for particle in particles:
            weight = weightStart - ((weightStart - weightEnd) * i)/iterations
            for crd in range(len(particle.coordinates)):
                velocity = GetVelocity(c1,c2, weight, particle.velocity[crd], particle.coordinates[crd], pBest.coordinates[crd], gBest.coordinates[crd])
                if(velocity > initialVelocity):
                     velocity = initialVelocity
                if(velocity < -initialVelocity):
                    velocity = -initialVelocity
                particle.velocity[crd] = velocity
                newVal = particle.coordinates[crd] + velocity
                particle.coordinates[crd] = newVal
                if(particle.coordinates[crd]< def_min or particle.coordinates[crd] > def_max):
                    particle.coordinates[crd] = random.uniform(def_min,def_max)
            particle.setZ(GetFunction(particle.coordinates, func))
            if(particle.z < pBest.z):
                pBest = particle
            ax.scatter(particle.coordinates[0],particle.coordinates[1],particle.z-0.2,color="b",s=20)
        
            if(pBest.z < gBest.z):
                gBest = pBest
        ax.scatter(gBest.coordinates[0],gBest.coordinates[1],gBest.z-0.2,color="r",s=20)
        plt.pause(0.1)
        
        ax.clear()
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    plt.show()

def HillClimbAlghorithm(startCoordinates, func):  

    initIndividualPoint = IndividualPoint(startCoordinates)
    initIndividualPoint.setZ(GetFunction(initIndividualPoint.coordinates, func))
    globalMinFound = False
    
    #print("drawing plot ... ")

    fig = plt.figure()
    ax = fig.gca( projection='3d')
    x = y = np.arange(def_min, def_max, stepDrawing)
    X, Y = np.meshgrid(x, y)
    zs = np.array([GetFunction((x,y), func) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    max_iterations = 500
    counter = 0

    while globalMinFound == False:
        counter +=1
        # generate neighbourhood
        individualPoints = GenerateIndividualPointsInNeighbourhood(step, initIndividualPoint)
        
        # calculate z
        for i in range(len(individualPoints)):
            individualPoints[i].setZ( GetFunction(individualPoints[i].coordinates, func))

        #find minimum    
        lowestIndividualPoint = FindGlobalMininum(individualPoints)    

        # draw point    
        ax.scatter(lowestIndividualPoint.coordinates[0],lowestIndividualPoint.coordinates[1],lowestIndividualPoint.z,color="k",s=20)
        # if prev individualPoint == curresnt individualPoint => we found minimum
        if(max_iterations < counter or (initIndividualPoint.coordinates[0] == lowestIndividualPoint.coordinates[0] and initIndividualPoint.coordinates[1] == lowestIndividualPoint.coordinates[1])):
            globalMinFound = True            
            ax.scatter(lowestIndividualPoint.coordinates[0],lowestIndividualPoint.coordinates[1],lowestIndividualPoint.z,color="r",s=40)
        else:
            initIndividualPoint = lowestIndividualPoint
            #print (lowestIndividualPoint.z)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    #plt.show()
    #print("Plot draw  finished")
    return lowestIndividualPoint.z



def BlindAlghorithm(startCoordinates, iterations, func):  

    lowestIndividualPoint = IndividualPoint(startCoordinates)
    lowestIndividualPoint.setZ(GetFunction(lowestIndividualPoint.coordinates, func))
    
    print("drawing plot ... ")

    fig = plt.figure()
    ax = fig.gca( projection='3d')
    x = y = np.arange(def_min, def_max, stepDrawing)
    X, Y = np.meshgrid(x, y)
    zs = np.array([GetFunction((x,y), func) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    for i in range(iterations):
        # generate neighbourhood
        newIndividualPoint = GenerateRandomIndividualPoint(def_min, def_max)
        
        # calculate z
        newIndividualPoint.setZ( GetFunction(newIndividualPoint.coordinates, func))

        # draw point    
        ax.scatter(newIndividualPoint.coordinates[0],newIndividualPoint.coordinates[1],newIndividualPoint.z,color="k",s=20)
        print (newIndividualPoint.z)

        # if prev individualPoint == current individualPoint => we found minimum
        if(lowestIndividualPoint.z > newIndividualPoint.z):
            lowestIndividualPoint = newIndividualPoint

    ax.scatter(lowestIndividualPoint.coordinates[0],lowestIndividualPoint.coordinates[1],lowestIndividualPoint.z,color="r",s=40)
    print ("Finnaly found: ")
    print (lowestIndividualPoint.z)


    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    print("Plot draw  finished")
    return lowestIndividualPoint.z

def DiffetentialAlgorithm(defMin, defMax, func):
    F = 0.7
    CR = 0.75
    popLength = 20
    generations = 40
    dim = 2
    population = []
    newPopulation = []
    for i in range(popLength):
        pt = GenerateRandomIndividualPoint(defMin, defMax)
        population.append(pt)
        newPopulation.append(pt)
        tmpZ = GetFunction(population[i].coordinates, func)
        population[i].setZ(tmpZ)
        newPopulation[i].setZ(tmpZ)

    minimum = FindGlobalMininum(population)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(def_min, def_max, stepDrawing)
    X, Y = np.meshgrid(x, y)
    zs = np.array([GetFunction((x,y), func) for x,y in zip(np.ravel(X), np.ravel(Y))])  
    Z = zs.reshape(X.shape)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap='viridis', edgecolor='none')

    for gen in range(generations):
        ax.clear()
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
        for i in range(popLength):
            finalCoords = []
            #newPoint=  IndividualPoint([0,0])
            diffVector = GetDifferentialVector(population[i])
            noise = CalculateNoise(diffVector[0], diffVector[1], diffVector[2], F, dim)
            #noise = CalculateNoiseCurrentToBest(population[i], minimum, diffVector[0], diffVector[1], F, dim)
            for coor in range(dim):
                rNumber = random.uniform(0,1)
                if (rNumber < CR):
                    #newPoint.coordinates[coor] = noise.coordinates[coor]
                    finalCoords.append(noise.coordinates[coor])
                else:
                    #newPoint.coordinates[coor] = population[i].coordinates[coor]
                    finalCoords.append(population[i].coordinates[coor])
            tmpZ = GetFunction(finalCoords, func)
            if(tmpZ < newPopulation[i].z):
                newPopulation[i].z = tmpZ
                newPopulation[i].coordinates = finalCoords
            minimum = FindGlobalMininum(newPopulation)
            ax.scatter(newPopulation[i].coordinates[0], newPopulation[i].coordinates[1], newPopulation[i].z-0.2, color="b", s=20)
            #ax.scatter(minimum.coordinates[0], minimum.coordinates[1], minimum.z-0.2, color="r", s=20)
            #plt.AddScatters(ax, population, "r")
        plt.pause(0.05)
    plt.show()
        

def CalculateNoise(p1,p2,p3, F, dim):
    coords = []
    for i in range(dim):
        coords.append(p1.coordinates[i] + F * (p2.coordinates[i] - p3.coordinates[i])),
    point = IndividualPoint(coords)
    return point


def CalculateNoiseCurrentToBest(pActual, pBest, p1, p2, F, dim):
    lam = 0.6
    coords = []
    for i in range(dim):
        coords.append( pActual.coordinates[i] + lam * (pBest.coordinates[i] - pActual.coordinates[i]) + F * (p1.coordinates[i] - p2.coordinates[i])),
    point = IndividualPoint(coords)
    return point


def GetDifferentialVector(individual):
    points = []
    for i in range(3):
        equality = True
        while equality:
            equality = False
            currentPoint = GenerateRandomIndividualPoint(def_min, def_max)
            if(individual.coordinates[0] == currentPoint.coordinates[0] and individual.coordinates[1] == currentPoint.coordinates[1]):
                equality = True
            for i in range(len(points)):
                if(points[i].coordinates[0] == currentPoint.coordinates[0] and points[i].coordinates[1] == currentPoint.coordinates[1]):
                    equality = True
        points.append(currentPoint)
    return points

def SimulatedAnnealingAlghorithm(startCoordinates, func, printing = False, draw = False):
    temperature = 50000
    tf= 0.00001
    lamb = 0.99


    lowestIndividualPoint = IndividualPoint(startCoordinates)
    lowestIndividualPoint.setZ(GetFunction(lowestIndividualPoint.coordinates, func))
    
    #print("drawing plot ... ")

    fig = plt.figure()
    ax = fig.gca( projection='3d')
    x = y = np.arange(def_min, def_max, stepDrawing)
    X, Y = np.meshgrid(x, y)
    zs = np.array([GetFunction((x,y), func) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    while(temperature > tf):
        # generate neighbourhood point
        newIndividualPoint = GenerateRandomIndividualPointForSA(lowestIndividualPoint)
        
        # calculate z
        newIndividualPoint.setZ( GetFunction(newIndividualPoint.coordinates, func))

        # draw point
        if(draw == True):  
            ax.scatter(newIndividualPoint.coordinates[0],newIndividualPoint.coordinates[1],newIndividualPoint.z,color="k",s=20)
        if(printing == True):
            print (newIndividualPoint.z)

        if(lowestIndividualPoint.z > newIndividualPoint.z):
            lowestIndividualPoint = newIndividualPoint
        else:
            delta = newIndividualPoint.z - lowestIndividualPoint.z
            r = random.uniform(0,1)
            if(r < m.exp(-delta/temperature)):                
                lowestIndividualPoint = newIndividualPoint
        temperature *= lamb


    if(draw == True):
        ax.scatter(lowestIndividualPoint.coordinates[0],lowestIndividualPoint.coordinates[1],lowestIndividualPoint.z,color="r",s=40)

    if(printing == True):
        print ("Simulated Annealing finnaly found: ")
        print (lowestIndividualPoint.z)


    if(draw == True):
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
    return lowestIndividualPoint.z



startPoint = (-15,10)

def_min = -50
def_max =  50
step = 1
stepDrawing = 1



#  'sphere'
#  'rastrigin'
#  'rosenbrock'
#  'ackley'
#  'schwefel'

#HillClimbAlghorithm(startPoint, 'schwefel')



#SimulatedAnnealingAlghorithm(startPoint, 'schwefel', True, True)

# count = 5
# totalSA = 0
# totalHC = 0
# print ("Started simulated annealing  for schwefel function: ")
# for i in range(count):    
#     tmpSA_schwefel = SimulatedAnnealingAlghorithm(startPoint, 'schwefel')
#     tmpHC_schwefel = HillClimbAlghorithm(startPoint, 'schwefel')
#     print ("SA:\t" + str(tmpSA_schwefel))
#     print("HC:\t"  + str(tmpHC_schwefel))
#     print("")
#     totalSA += tmpSA_schwefel
#     totalHC += tmpHC_schwefel
# averageSA = totalSA/count
# averageHC = totalHC/count

# print ("Average Simulated annealing:")
# print ( averageSA)

# print ("Average hill climb:")
# print ( averageHC)


# totalHC = 0
# print ("Started hill climb : ")
# for i in range(count):    
#     tmp = HillClimbAlghorithm(startPoint, 'schwefel')
#     print (tmp)
#     totalHC += tmp
# averageHC = totalHC/count

# print ("Average hill climb:")
# print ( averageHC)


#BlindAlghorithm(startPoint,100, 'schwefel')



startPoint = (-40,-40)

#HillClimbAlghorithm(startPoint, 'sphere')
#BlindAlghorithm(startPoint,100, 'sphere')




# startPoint = (-5,5)
# step = 0.1
# stepDrawing = 0.1
# def_min = -5
# def_max =  5

#HillClimbAlghorithm(startPoint, 'rastrigin')
#SimulatedAnnealingAlghorithm(startPoint, 'rastrigin')
# BlindAlghorithm(startPoint, 100, 'rastrigin')




startPoint = (2,-2)
step = 0.05
stepDrawing = 0.1
def_min = -3
def_max =  3

#HillClimbAlghorithm(startPoint, 'rosenbrock')
#SimulatedAnnealingAlghorithm(startPoint, 'rosenbrock')
# BlindAlghorithm(startPoint, 100, 'rosenbrock')



# startPoint = (2,-2)
# step = 0.1
# stepDrawing = 0.1
# def_min = -5
# def_max =  5

# HillClimbAlghorithm(startPoint, 'ackley')
# BlindAlghorithm(startPoint, 100, 'ackley')

step = 0.05
stepDrawing = 1
def_min = -30
def_max =  30
#SomaAlgorithm('ackley')
#ParticleSwarm('sphere',0.2,0.2)

DiffetentialAlgorithm(def_min,def_max, 'sphere')
