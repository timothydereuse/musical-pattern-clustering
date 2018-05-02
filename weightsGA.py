
import random

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

def runGA(inst, numAttr, evaluate):

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Structure initializers - 
    # not sure how to use this without a tools.init function
    toolbox.register("individual", tools.initRepeat, creator.Individual, inst, numAttr)
    
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    #def evalOneMax(individual):
    #    return sum(individual),
    
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    random.seed(64)
    
    pop = toolbox.population(n=200)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=30, 
                                   stats=stats, halloffame=hof, verbose=True)
    
    return pop, log, hof

#if __name__ == "__main__":
#    main()