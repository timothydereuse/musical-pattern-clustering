import random
import datetime
import pickle
import numpy
import featureExtractors as ft
import patternClass as pc
from collections import Counter
from deap import base
from deap import creator
from deap import tools
import functools

def main():

    print("loading data from file...")
    with open('parsed_patterns.pik', "rb") as f:
        dat = pickle.load(f)

    songs = dat[0]
    pClasses = dat[1]
    pOccs = dat[2]
    annPClassNames = dat[3]
    annPOccNames = dat[4]
    genPClassNames = dat[5]
    genPOccNames = dat[6]
    filtGenPClassNames = dat[7]

    pClassFeatureKeys = pClasses[annPClassNames[0]].classFeatures.keys()
    pClassFeatureKeys = sorted(pClassFeatureKeys)
    num_chunks = 5

    numpy.random.shuffle(annPClassNames)
    numpy.random.shuffle(filtGenPClassNames)

    ann_chunks = split_into_chunks(annPClassNames,num_chunks)
    gen_chunks = split_into_chunks(filtGenPClassNames,num_chunks)
    data_sets = [ann_chunks[i] + gen_chunks[i] for i in range(num_chunks)]

    def instAttribute():
        return random.uniform(0,1)

    subset = keys_subset(pClassFeatureKeys,'exclude_stds')
    numAttributes = len(subset)
    defaultWeights = numAttributes * [1]

    for num_run in range(num_chunks):

        testPClassNames = data_sets[num_run]
        valPClassNames = [data_sets[i] for i in range(num_chunks) if i is not num_run]
        valPClassNames = [item for sublist in valPClassNames for item in sublist]

        validateWeights = functools.partial(performKNNwithLOOCV,
                valPatternClassNames = valPClassNames,
                kNearest = 10,
                patternClasses = pClasses,
                useKeys = subset
                )

        testWeights = functools.partial(performKNN,
                valPatternClassNames = testPClassNames,
                trainPatternClassNames = valPClassNames,
                kNearest = 10,
                patternClasses = pClasses,
                useKeys = subset
                )

        #print(testWeights(defaultWeights))

        #file to write results into
        currentTime = str(datetime.datetime.now())
        filename = "GA DATA num " + str(num_run) + " at " + currentTime + ".txt"
        filename = filename.replace(":","-")

        runGA(instAttribute,numAttributes,validateWeights,testWeights,filename)

    pass

def keys_subset(all_keys,type_string):
    if type_string == 'only_pitch':
        return [x for x in all_keys if ('pitch' in x or 'interval' in x)]
    elif type_string == 'only_rhythm':
        return [x for x in all_keys if ('rhythm' in x)]
    elif type_string == 'exclude_means':
        return [x for x in all_keys if ('avg' not in x)]
    elif type_string == 'exclude_stds':
        return [x for x in all_keys if ('std' not in x)]
    elif type_string == 'exclude_song_comp':
        return [x for x in all_keys if ('diff' not in x)]
    else:
        return all_keys
    pass

def split_into_chunks(inp,num_chunks):

    chunk_len = int(numpy.floor(len(inp) / num_chunks))
    chunks = [inp[i:i + chunk_len] for i in range(0, len(inp), chunk_len)]
    if len(chunks) > num_chunks:
        for i,x in enumerate(chunks[num_chunks]):
            chunks[i].append(x)
        del chunks[num_chunks]

    return chunks

def performKNN(weights, kNearest, trainPatternClassNames, valPatternClassNames,
               patternClasses, useKeys = None):
    """
    returns a float representing the proportion of correctly classified patternClasses
    using KNN with the settings given
    patternClasses: master list of patternClasses
    trainPatternClassNames: list of names to train with
    valPatternClassNames: list of names to validate with
    weights: weights to be tested
    kNearest: k for knn search
    """
    #start = timer()
    if useKeys == None:
        sortKeys = sorted(patternClasses[valPatternClassNames[0]].classFeatures.keys())
    else:
        sortKeys = sorted(useKeys)

    correctClass = 0;

    #now time to test some patternClasses!
    for tstMtf in valPatternClassNames:
        curPatternClass = patternClasses[tstMtf]
        curFeats = curPatternClass.classFeatures

        #distances links names of patterns in the training set to their
        #distance to the current item. will never be larger than k.
        distances = {}
        maxDistValue = 100000000

        #get distance from this testpatternClass to all trainpatternClasses
        #this loop performs 1 KNN search for every iteration
        for tm in trainPatternClassNames:

            trainFeats = patternClasses[tm].classFeatures
            #the innermost loop:
            dist = 0
            ind = 0

            for ind in range(0,len(sortKeys)):
                #well, if i'm just doing feature selection, then i can get rid
                #of the multiplication in the weights...
                if weights[ind] == 0:
                    continue
                curKey = sortKeys[ind]
                temp = (curFeats[curKey] - trainFeats[curKey]) * weights[ind]
                dist += temp * temp
                if dist > maxDistValue:
                    break

            #if we don't yet have a full distances dict yet, then just add this
            #to the list. otherwise, check if it's in the closest k so far; if
            #it is, then remove the current highest-distance entry in distances
            #and then add this one back in

            if len(distances) < kNearest:
                distances[tm] = dist
            elif dist < maxDistValue:
                keyToRemove = max(distances,key=distances.get)
                del distances[keyToRemove]
                distances[tm] = dist

            maxDistValue = max(distances.values())

        sortDistNames = sorted(distances, key=distances.get)

        closestpatternClassTypes = [patternClasses[mn].type for mn in sortDistNames]
        countTypes = Counter(closestpatternClassTypes)
        predictedClass = sorted(countTypes, key=countTypes.get)[-1]

        if(predictedClass == curPatternClass.type):
            correctClass +=1
    #end = timer()
    #print(end - start)

    return correctClass / len(valPatternClassNames)

def performKNNwithLOOCV(weights, valPatternClassNames, kNearest, patternClasses, useKeys=None):
    """
    a wrapper function for performKNN that performs leave-one-out
    cross-validation.
    """
    correctRuns = 0;

    for i in range(len(valPatternClassNames)):
        trainNames = valPatternClassNames[:i] + valPatternClassNames[(i+1):]
        valNames = [valPatternClassNames[i]]
        #(valNames)

        res = performKNN(weights,kNearest,trainNames,valNames,
                            patternClasses,useKeys)
        correctRuns += res
        #print(res)


    return correctRuns / len(valPatternClassNames)

def runGA(inst, numAttr, evaluate, test_func, filename = None):

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual,
        inst,numAttr)
    # define the population to be a list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    #----------
    # Operator registration
    #----------
    # register the goal / fitness function
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxUniform)

    # register a mutation operator
    toolbox.register("mutate", tools.mutPolynomialBounded, eta=0.4, low=0, up=1, indpb=0.06)
    #toolbox.register("mutate", tools.mutUniformInt, low=0, up=32, indpb=0.06)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    toolbox.register("select", tools.selTournament, tournsize=3)
    #random.seed(64)

    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=50)

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.5, 0.4

    print("Start of evolution")

    # Evaluate the entire population
    fitnesses = [(toolbox.evaluate(weights=x),) for x in pop]
    #fitnesses = toolbox.map(toolbox.evaluate,pop)

    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0

    #array keeping track of fittnesses
    genFitArr = []

    continue_evolving = True

    # Begin the evolution
    while g < 10000 and continue_evolving:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2, 0.5)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        #fitnesses = map(toolbox.evaluate, invalid_ind)
        fitnesses = [(toolbox.evaluate(weights=x),) for x in invalid_ind]
        #fitnesses = parallel(delayed(toolbox.evaluate)(weights=x) for x in invalid_ind)


        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        #print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        #print("  Avg %s" % mean)
        #print("  Std %s" % std)

        genFitArr.append(round(mean,5))
        if( g > 5 and (max(genFitArr[-5:-1]) - min(genFitArr[-5:-1])) < 0.005 ):
            print('convergence reached - halting evolution')
            continue_evolving = False

        best_ind = tools.selBest(pop, 1)[0]
        print(str([round(x,2) for x in best_ind]) + "\n")
        print(" test set %s " % test_func(best_ind))
        print(genFitArr)

        file = open(filename,"a")
        file.write("  GEN. " + str(g))
        file.write("  Min %s" % round(min(fits),6))
        file.write("  Max %s" % round(max(fits),6))
        file.write("  Avg %s" % round(mean,6))
        file.write("  Std %s" % round(std,6))
        file.write(str(best_ind))
        file.write("\n ")
        file.close()

    print("-- End of evolution --")

    best_ind = tools.selBest(pop, 1)[0]

    file = open(filename,"a")
    file.write("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    file.write(str(print(genFitArr)))
    file.write(" test set: %s " % test_func(best_ind))
    file.close()



if __name__ == "__main__":
    main()
