
import random
import datetime
import pickle
import numpy as np
import sys
import time
import featureExtractors as ft
import patternClass as pc
from collections import Counter
from deap import base
from deap import creator
from deap import tools
from timeit import default_timer as timer
import functools
from scoop import futures
#from multiprocessing import Pool

from importlib import reload

reload(pc)
reload(ft)

# def keys_subset(all_keys,type_string):
#     if type_string == 'only_pitch':
#         return [x for x in all_keys if ('pitch' in x or 'interval' in x)]
#     elif type_string == 'only_rhythm':
#         return [x for x in all_keys if ('rhythm' in x)]
#     elif type_string == 'exclude_means':
#         return [x for x in all_keys if ('avg' not in x)]
#     elif type_string == 'exclude_stds':
#         return [x for x in all_keys if ('std' not in x)]
#     elif type_string == 'exclude_song_comp':
#         return [x for x in all_keys if ('diff' not in x and 'expected' not in x)]
#     elif type_string == 'all':
#         return all_keys
#     else:
#         raise TypeError('bad keys_subset type ' + str(type_string))
#     pass
#
# def split_into_chunks(inp,num_chunks):
#
#     chunk_len = int(np.floor(len(inp) / num_chunks))
#     chunks = [inp[i:i + chunk_len] for i in range(0, len(inp), chunk_len)]
#     if len(chunks) > num_chunks:
#         for i,x in enumerate(chunks[num_chunks]):
#             chunks[i].append(x)
#         del chunks[num_chunks]
#
#     return chunks

def perform_knn(weights, kNearest, trainPatternClassNames, valPatternClassNames,
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

def perform_knn_with_loocv(weights, valPatternClassNames, kNearest,
        patternClasses, useKeys=None, run_fraction = 1):
    """
    a wrapper function for performKNN that performs leave-one-out
    cross-validation.
    """
    correctRuns = 0;

    indices_to_test = list(range(len(valPatternClassNames)))
    num_tests = round(run_fraction * len(valPatternClassNames))
    if run_fraction < 1:
        indices_to_test = random.sample(indices_to_test,num_tests)

    for i in indices_to_test:
        trainNames = valPatternClassNames[:i] + valPatternClassNames[(i+1):]
        valNames = [valPatternClassNames[i]]

        res = perform_knn(weights,kNearest,trainNames,valNames,
                            patternClasses,useKeys)
        correctRuns += res
        #print(res)


    return correctRuns / num_tests

def test_subsets(k_vals,all_keys,names,classes):

    subsets = ['only_pitch','only_rhythm','exclude_means',
            'exclude_stds','exclude_song_comp','all']
    res = str(k_vals) + "\n"

    for s in subsets:
        this_keys = keys_subset(all_keys,s)
        weights = [1]*len(this_keys)
        res += s + ' '
        for k in k_vals:
            t = perform_knn_with_loocv(weights,names,k,classes,this_keys)
            res += "& " + str(round(t*100,1)) + " "
        res += "\n"

    return res

def runGA(num_run, ga_population, mutation_prob, k_nearest, feature_subset,
    data_sets, pClasses, time_limit = -1, convergence_thresh = 0.001):
    """
    num_run: integer <= num_chunks, says which member of data_sets to use for
        testing. the rest will be concatenated used for training
    ga_population: population to use for the GA
    mutation_prob: probability for an individual to undergo mutation
    k_nearest: value of K to use in the KNN
    feature_subset: feature category to operate on
    data_sets: list of lists containing data sets split into test and train
    pClasses: master pattern classes variable
    time_limit: time in seconds to run this test for; ends on first generation
        that finishes testing after the time specified has elapsed
    convergence_thresh: tolerance before convergence assumed
    """

    # CXPB = CROSSOVER LIKELIHOOD
    CXPB = 0.5

    #setup
    pClassFeatureKeys = list(pClasses[list(pClasses.keys())[0]].classFeatures.keys())
    subset = ft.keys_subset(pClassFeatureKeys,feature_subset)
    num_attr = len(subset)

    instAttribute = functools.partial(random.uniform,0,1)

    num_chunks = len(data_sets)
    test_pat_names = data_sets[num_run]
    valPClassNames = [data_sets[i] for i in range(num_chunks) if i is not num_run]
    valPClassNames = [item for sublist in valPClassNames for item in sublist]

    evaluate = functools.partial(perform_knn_with_loocv,
            valPatternClassNames = valPClassNames,
            kNearest = k_nearest,
            patternClasses = pClasses,
            useKeys = subset,
            run_fraction = 0.7
            )

    test_func = functools.partial(perform_knn,
            valPatternClassNames = test_pat_names,
            trainPatternClassNames = valPClassNames,
            kNearest = k_nearest,
            patternClasses = pClasses,
            useKeys = subset
            )

    #file to write results into
    currentTime = str(datetime.datetime.now())
    filename = "GA DATA num " + str(num_run) + " at " + currentTime + ".txt"
    filename = "GA DATA %s,%s,%s.txt" % (num_run,feature_subset,currentTime)
    filename = filename.replace(":","-")

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()

    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual,
        instAttribute,num_attr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    #----------
    # Operator registration
    #----------
    # register the goal / fitness function
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxUniform)

    # register a mutation operator
    toolbox.register("mutate", tools.mutPolynomialBounded, eta=5, low=0, up=1, indpb=1)
    #toolbox.register("mutate", tools.mutUniformInt, low=0, up=32, indpb=0.06)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    toolbox.register("select", tools.selTournament, tournsize=3)
    #random.seed(64)

    # create an initial population of ga_population individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=ga_population)


    print("Start of evolution")
    # Evaluate the entire population
    fitnesses = [(toolbox.evaluate(weights=x),) for x in pop]
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Extracting all fitnesses
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0

    #array keeping track of fittnesses
    genFitArr = []

    continue_evolving = True
    start_time = time.time()

    # Begin the evolution
    while g < 10000 and continue_evolving:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2, 0.5)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        # Apply mutation on the offspring
        for mutant in offspring:

            # mutate an individual with probability mutation_prob
            if random.random() < mutation_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals whose fitness is not known
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = [(toolbox.evaluate(weights=x),) for x in invalid_ind]

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("Evaluated %i individuals" % len(invalid_ind))

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

        #the GA converges if:
        #- at least 8 generations have passed
        #- the fitness difference between the best generation and the 10th
        #  best is  below a threshold of convergence_thresh
        if( g > 8 ):
            best = sorted(genFitArr)[-8:]
            if (best[-1] - best[0] < convergence_thresh):
                print('convergence reached - halting evolution')
                continue_evolving = False

        #if we've gone past the time limit, then stop evolving
        elapsed_time = time.time() - start_time
        if(elapsed_time > time_limit):
            print('time limit reached - halting evolution')
            continue_evolving = False

        best_ind = tools.selBest(pop, 1)[0]
        print(str([round(x,2) for x in best_ind]) + "\n")
        #print("test set %s " % test_func(best_ind))
        #print(str(genFitArr) + "\n")

        file = open(filename,"a")
        file.write("  GEN. " + str(g))
        file.write("  Min %s" % round(min(fits),4))
        file.write("  Max %s" % round(max(fits),4))
        file.write("  Avg %s" % round(mean,4))
        file.write("  Std %s" % round(std,4))
        file.write("\n ")
        file.close()

    print("-- End of evolution --")

    best_ind = tools.selBest(pop, 1)[0]

    file = open(filename,"a")
    file.write("Best individual is %s, %s \n" % (best_ind, best_ind.fitness.values))
    file.write(str(genFitArr) + "\n")
    file.write("test set: %s \n " % test_func(best_ind))
    file.close()

def loocv_testing(k_vals,subset_name):

    keys = keys_subset(pClassFeatureKeys,subset_name)
    defaultWeights = len(keys) * [1]
    s = ''
    for k in k_vals:
        res = perform_knn_with_loocv(defaultWeights,annPClassNames+filtGenPClassNames,k,pClasses,keys)
        s += "& " + str(round(res*100,1)) + " "
        #print('result for ' + str(k) + ': ' + str(round(res*100,1)))
    return s

if __name__ == "__main__":
    __spec__ = None

    num_chunks = 5
    ga_population = 25
    mutation_prob  = 0.01
    feature_subset = 'all'
    k_nearest = 15

    if (len(sys.argv) > 1):
        feature_subset = sys.argv[1]

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

    np.random.shuffle(annPClassNames)
    np.random.shuffle(filtGenPClassNames)

    ann_chunks = ft.split_into_chunks(annPClassNames,num_chunks)
    gen_chunks = ft.split_into_chunks(filtGenPClassNames,num_chunks)
    data_sets = [ann_chunks[i] + gen_chunks[i] for i in range(num_chunks)]

    partial_ga = functools.partial(runGA,
        ga_population=ga_population,
        mutation_prob=mutation_prob,
        feature_subset=feature_subset,
        k_nearest=k_nearest,
        data_sets=data_sets,
        pClasses=pClasses
        )

    print(partial_ga(num_run=3,time_limit=60*10))
    #with Pool(3) as p:
    #    print(p.map(partial_ga,range(num_chunks)))
