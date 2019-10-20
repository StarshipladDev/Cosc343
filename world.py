#!/usr/bin/env python
#Author : Starshipladdev
from cosc343world import Creature, World
import numpy as np
import time
import matplotlib.pyplot as plt
import random

# You can change this number to specify how many generations creatures are going to evolve over...
numGenerations = 250
# You can change this number to specify how many turns in simulation of the world for given generation
numTurns = 100

# You can change this number to change the world type.  You have two choices - world 1 or 2 (described in
# the assignment 2 pdf document)
worldType = 2

# You can change this number to change the world size
gridSize = 35

# You can set this mode to True to have same initial conditions for each simulation in each generation.  Good
# for development, when you want to have some determinism in how the world runs from generatin to generation.
repeatableMode = False


# This is a class implementing the creature a.k.a MyCreature.  It extends the basic Creature, which provides the
# basic functionality of the creature for the world simulation.  Your job is to implement the AgentFunction
# that controls creature's behavoiur by producing actions in respons to percepts.
#

avgLife = []
Survivors = []
AverageLifeGraph = []
class MyCreature(Creature):
    # Initialisation function.  This is where your creature
    # should be initialised with a chromosome in random state.  You need to decide the format of your
    # chromosome and the model that it's going to give rise to
    #
    # Input: numPercepts - the size of percepts list that creature will receive in each turn
    #        numActions - the size of actions list that creature must create on each turn

    def __init__(self, numPercepts, numActions):

        # Place your initialisation code here.  Ideally this should set up the creature's chromosome
        # and set it to some random state.
        ##################
        ##################
        # OSCARS CODE FOR GENERATING RANDOM FIRST STATS
        ##################
        ##################

        self.chromoList = []
        self.actionList = []
        # 0 - Chance to do thing if energized
        # 1 - Chance to do thing if un-energized
        # 2-9 - What to do if Monster on [x]
        # 10-17 - What to do if Food on [x]
        # 18-What to do if currently on Red apple
        # 19- What to do if currently on Green Apple
        # 20-27 action to do if friend on [x]
        # 28 - Run away form monster weight
        # 29 Run away from friend weight
        # 30- Chance to do somthing if hungry
        # 31 Action and weight if multiple mosnters
        # 32 Action and weight if multiple food
        #33 weight if food
        i=0
        weight = .8
        while (i<34):
            self.chromoList.append([random.randint(0,10),random.uniform(0,weight)])
            i+=1
        i = 0
        while (i < 11):
            self.actionList.append(random.uniform(0, .2))
            i += 1
        #THE BELOW IS LEGACY 'PARAMETERS X OBJECTS' CHROMOSNES
        """
        i = 0
        while (i < 9):
            self.chromoList.append([[(random.randint(0, 10)) , random.uniform(0, .8)]])
            f = 0
            while (f < 2):
                self.chromoList[i].append([(random.randint(0, 10)), random.uniform(0, .8)])
                f += 1
            i += 1
        """
        # ENDOFLEGACYODE
        # Do not remove this line at the end.  It calls constructors
        # of the parent classes.
        Creature.__init__(self)

    # This is the implementation of the agent function that is called on every turn, giving your
    # creature a chance to perform an action.  You need to implement a model here, that takes its parameters
    # from the chromosome and it produces a set of actions from provided percepts
    #
    # Input: percepts - a list of percepts
    #        numAction - the size of the actions list that needs to be returned
	#GENERATE INITIAL STATS
    def FitnessFunction(self):
        score = self.getEnergy()/4 + self.timeOfDeath()
        if self.isDead()==False:
            score+=25
        return score

    def AgentFunction(self, percepts, numActions):

        # At the moment the actions is a list of random numbers.  You need to
        # replace this with some model that maps percepts to actions.  The model
        # should be parametrised by the chromosome
        # actions = np.random.uniform(0, 1, size=numActions)
        #ACTIONLIST BELOW
        # 3 food 2 other 1 mosnter 0 empty
        # 2 red 1 green 0 no food

        # 0 - Chance to do thing if energized
        # 1 - Chance to do thing if un-energized
        # 2-9 - What to do if Monster on [x]
        # 10-17 - What to do if Food on [x]
        # 18-What to do if currently on Red apple
        # 19- What to do if currently on Green Apple
        # 20-27 action to do if friend on [x]
        # 28 - Run away form monster weight
        # 29 Run away from friend weight
        # 30- Chance to do somthing if hungry
        # 31 Action and weight if multiple mosnters
        # 32 Action and weight if multiple food
        # 33 Weight if food
        MonsterNum = 0
        FoodNum = 0
        actions = self.actionList[:]
        i = 0
        while(i<len(percepts)):
            thing = int(percepts[i])
            if(i==4):
                if(thing==2):
                    actions[9] += self.chromoList[18][1]
                elif(thing==1):
                    actions[9]+=self.chromoList[19][1]
            elif(thing==3):
                FoodNum+=1
                if(i<4):
                    actions[self.chromoList[i+10][0]] += self.chromoList[33][1]
                else:
                    actions[self.chromoList[i+9][0]] += self.chromoList[33][1]
            elif(thing==1):
                MonsterNum+=1
                if (i < 4):
                    actions[self.chromoList[i + 2][0]] += self.chromoList[28][1]
                else:
                    actions[self.chromoList[i + 1][0]] += self.chromoList[28][1]
            elif (thing == 2):
                if (i < 4):
                    actions[self.chromoList[i + 20][0]] += self.chromoList[29][1]
                else:
                    actions[self.chromoList[i + 19][0]] += self.chromoList[29][1]
            i += 1

        #LEGACY CODE FOR MULTIPLE OBJECTS
        #if(MonsterNum>1):
            #actions[self.chromoList[31][0]]+= self.chromoList[31][1]
        #if(FoodNum>1):
            #actions[self.chromoList[32][0]] += self.chromoList[32][1]
        #LEGACY CODE FOR PERCEPTS X OBJECTS AGENT FUNCTION
            """
        i = 0
        while (i < len(percepts)):
            if (percepts[i] != 0):
                actions[self.chromoList[i][int(percepts[i] - 1)][0]] += self.chromoList[i][int(percepts[i] - 1)][1]
            i += 1
            """
        return actions
        # ENDOFLEGACYCODE

    ##################
    ##################
    #CODE FOR RETREIVING AND SETTING CHROMOSONES
    ##################
    ##################
    def GetActionList(self):
        return self.actionList

    def SetActionList(self, al, al1, mutationConstant):
        y = []
        i = 0
        while (i < len(al)):
            if (random.uniform(0, 1) > .5):
                y.append(al[i])
            else:
                y.append(al1[i])
            i += 1
        self.actionList = y
        i = 0
        while (i < mutationConstant):
            r = random.randint(0, 10)
            self.actionList[r] = (random.uniform(0, .2))
            i += 1

    def GetchromoList(self):
        return self.chromoList

    def SetchromoList(self, al, al1, mutationConstant):
        y = []
        i = 0
        while (i < len(al)):
            if (random.uniform(0, 1) > .5):
                y.append(al[i])
            else:
                y.append(al1[i])
            i += 1
        self.chromoList = y
        i = 0
        f = 0
        while (i < mutationConstant):
            if(random.randint(0,100)<5):
                r = random.randint(0,33)
                self.chromoList[r]=[(random.randint(0, 10)), random.uniform(0, .8)]
            i += 1


# This function is called after every simulation, passing a list of the old population of creatures, whose fitness
# you need to evaluate and whose chromosomes you can use to create new creatures.
#
# Input: old_population - list of objects of MyCreature type that participated in the last simulation.  You
#                         can query the state of the creatures by using some built-in methods as well as any methods
#                         you decide to add to MyCreature class.  The length of the list is the size of
#                         the population.  You need to generate a new population of the same size.  Creatures from
#                         old population can be used in the new population - simulation will reset them to starting
#                         state.
#
# Returns: a list of MyCreature objects of the same length as the old_population.


def newPopulation(old_population):
    global numTurns
    survivorList = []
    nSurvivors = 0
    avgLifeTime = 0
    fitnessScore = 0
    ###
    # OSCARSCODE
    ####

    fitnessTotal = 0

    ####
    # For each individual you can extract the following information left over
    # from evaluation to let you figure out how well individual did in the
    # simulation of the world: whether the creature is dead or not, how much
    # energy did the creature have a the end of simualation (0 if dead), tick number
    # of creature's death (if dead).  You should use this information to build
    # a fitness function, score for how the individual did
    r = 0
    for individual in old_population:

        # You can read the creature's energy at the end of the simulation.  It will be 0 if creature is dead
        energy = individual.getEnergy()

        # This method tells you if the creature died during the simulation
        dead = individual.isDead()

        # If the creature is dead, you can get its time of death (in turns)
        if dead:
            timeOfDeath = individual.timeOfDeath()
            avgLifeTime += timeOfDeath
        else:
            nSurvivors += 1
            avgLifeTime += numTurns
        ##################
        ##################
        #CODE FOR PLOTTING FITTEST
        ##################
        ##################

        AverageLife = 0
        fitnessTotal +=individual.FitnessFunction()
        AverageLife += avgLifeTime
    avgLife.append(fitnessTotal / len(old_population))
    AverageLifeGraph.append(AverageLife / len(old_population))
    # ENDOFCODE
    # Here are some statistics, which you may or may not find useful

    avgLifeTime = float(avgLifeTime) / float(len(population))
    # avgLife.append(avgLifeTime)
    Survivors.append(nSurvivors)
    print("Simulation stats:")
    print("  Survivors    : %d out of %d" % (nSurvivors, len(population)))
    print("  Avg life time: %.1f turns" % avgLifeTime)

    # The information gathered above should allow you to build a fitness function that evaluates fitness of
    # every creature.  You should show the average fitness, but also use the fitness for selecting parents and
    # creating new creatures.

    # Based on the fitness you should select individuals for reproduction and create a
    # new population.  At the moment this is not done, and the same population with the same number
    # of individuals
    ##################
    ##################
    #CODE FOR FINDING FITTEST POPUALTION AND ADDING 5 RANDOMS TO POP POOL
    ##################
    ##################
    EliteNumbers = 5
    BreedingSize = 20
    Mutants = 0
    MutantGenes = 6
    new_population = []
    i = 0
    elites = []
    # Get fittest 'E' individuals
    while (i < EliteNumbers):
        Swap = True
        new_population.append(MyCreature(numCreaturePercepts, numCreatureActions))
        f = 0
        fitestparent1 = old_population[random.randint(0, len(old_population) - 1)]
        while (f < len(old_population)):
            if (old_population[f].FitnessFunction()>fitestparent1.FitnessFunction()):
                for s in elites:
                    if s == f:
                        Swap = False
                    if Swap:
                        fitestparent1 = old_population[f]
                        elites.append(f)
            f += 1
        new_population[i].SetchromoList(fitestparent1.GetchromoList(), fitestparent1.GetchromoList(), 0)
        new_population[i].SetActionList(fitestparent1.GetActionList(), fitestparent1.GetActionList(), 0)
        i += 1
    #Fill rest of population
    while (i < len(old_population) - Mutants):
        new_population.append(MyCreature(numCreaturePercepts, numCreatureActions))
        fitnessSubset = random.sample(old_population, BreedingSize)
        f = 0
        parent1f=0
        fitestparent1 = fitnessSubset[0]
        fitestparent2 = fitnessSubset[1]
        while (f < len(fitnessSubset)):
            if (fitnessSubset[f].FitnessFunction() > fitestparent1.FitnessFunction()):
                parent1f=f
                fitestparent1 = fitnessSubset[f]
            f += 1
        f=0
        while (f < len(fitnessSubset)):
            if (fitnessSubset[f].FitnessFunction() > fitestparent2.FitnessFunction()):
                if(parent1f!=f):
                    fitestparent2 = fitnessSubset[f]
            f += 1
        new_population[i].SetchromoList(fitestparent1.GetchromoList(), fitestparent2.GetchromoList(), MutantGenes)
        new_population[i].SetActionList(fitestparent2.GetActionList(), fitestparent1.GetActionList(), MutantGenes)
        i += 1
    # Fill rest of population with 'K' mutants
    #FOr proper implementation, 'mutants' should be kept at 0
    while (i < len(old_population)):
        new_population.append(MyCreature(numCreaturePercepts, numCreatureActions))
        i += 1
    # ENDOFCODE
    # new_population = old_population
    return new_population


plt.close('all')
fh = plt.figure()

# Create the world.  Representaiton type choses the type of percept representation (there are three types to chose from);
# gridSize specifies the size of the world, repeatable parameter allows you to run the simulation in exactly same way.
w = World(worldType=worldType, gridSize=gridSize, repeatable=repeatableMode)

# Get the number of creatures in the world
numCreatures = w.maxNumCreatures()

# Get the number of creature percepts
numCreaturePercepts = w.numCreaturePercepts()

# Get the number of creature actions
numCreatureActions = w.numCreatureActions()

# Create a list of initial creatures - instantiations of the MyCreature class that you implemented
population = list()
for i in range(numCreatures):
    c = MyCreature(numCreaturePercepts, numCreatureActions)
    population.append(c)

# Pass the first population to the world simulator
w.setNextGeneration(population)

# Runs the simulation to evalute the first population
w.evaluate(numTurns)

# Show visualisation of initial creature behaviour
w.show_simulation(titleStr='Initial population', speed='fast')

for i in range(numGenerations):
    print("\nGeneration %d:" % (i + 1))

    # Create a new population from the old one
    population = newPopulation(population)

    # Pass the new population to the world simulator
    w.setNextGeneration(population)

    # Run the simulation again to evalute the next population
    w.evaluate(numTurns)

    # Show visualisation of final generation
    if i == numGenerations - 1:
        w.show_simulation(titleStr='Final population', speed='fast')
    ##################
    ##################
    #CODE FOR DISPALYING STATS
    ##################
    ##################
count = 0
i = 0
count = 0
while (i < len(avgLife)):
    count += avgLife[i]
    i += 1
print("Average Fitness %d " % (count / i))
i = 0
count = 0
while (i < len(Survivors)):
    count += Survivors[i]
    i += 1

print("Average Survivors %f " % (count / i))
#LEGACY CODE FOR MULIT PLOT GRAPHS
"""
plt.figure(1)
plt.subplot(311)
plt.plot(avgLife)
plt.xlabel("Generations")
plt.ylabel("Average Fitness")
plt.subplot(312)
plt.plot(Survivors)
plt.xlabel("Generations")
plt.ylabel("Number Of Survivors")
plt.subplot(313)
plt.plot(AverageLifeGraph)
"""
plt.xlabel("Generations")
plt.ylabel("Average Fitness")
plt.plot(avgLife)
plt.show()
"""
plt.plot(Survivorsplt)
plt.ylabel("Survivors")
plt.show()
"""

####
# END OF CODE
####