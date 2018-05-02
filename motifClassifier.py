# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 15:04:02 2017

@author: Tim
"""


#setup
#from music21 import *
import music21 as m21
#m21.environment.set('musescoreDirectPNGPath', 'C:\\Program Files (x86)\\MuseScore 2\\bin\\MuseScore.exe')
import os
import numpy
import csv
import featureExtractors
import random
import weightsGA

from collections import Counter

K_NEAREST = 3
PROP_TESTING = 0.15
PROP_VALIDATION = 0.15

#songNames -- unique IDs for each song
#motifNames -- unique IDs for each motif
#annList -- raw dump of annotation data
#songs -- dictionary containing the parsed score of each song, indexed by the
#        unique ID stored in songNames
#motifs -- dictionary containing the notes and position of each motif, indexed
#        by the unique ID stored in motifNames

#file directories
thisfileDir = os.path.dirname(os.path.realpath('__file__'))
fileDir = os.path.join(thisfileDir, '../MTC-ANN-2.0.1/krn')
annFile = os.path.join(thisfileDir, '../MTC-ANN-2.0.1/metadata/MTC-ANN-motifs.csv')

#get all the files into songNames
songNames = []

for root, dirs, files in os.walk(fileDir):
    for file in files:
        if file.endswith('.krn'):
            songNames.append(file[:-4])
    
#make dictionary linking filenames -> songs of each score
print("fetching scores...")
songs = {}
for f in songNames:
    songs[f] = {'score': m21.converter.parse(os.path.join(fileDir, f) + ".krn")}

# N.B. THE HEADERS ARE:
#0: tunefamily 
#1: songid
#2: motifid
#3: begintime
#4: endtime
#5: duration
#6: startindex
#7: endindex
#8: numberofnotes
#9: motifclass
#10: description	
#11: annotator
#12: changes
    
#now: get the annotations
print("fetching annotations...")
with open(annFile, 'r') as fp:
    reader = csv.reader(fp, delimiter=',', quotechar='"')
    annList = [row for row in reader]
    
print("extracting motifs...")
motifs = {}
motifNames = []
motifClasses = []
for entr in annList:
    motifs[entr[2]] = extractMotif(entr, songs)
    motifNames.append(entr[2])
    motifClasses.append(entr[9])
motifClasses = list(set(motifClasses))

#we want a dictionary linking each motif class to a list of motif names that
#occur in that class. this will be useful when splitting up the data for
#cross-validation
classToMotifs = {}
for mc in motifClasses:
    classToMotifs[mc] = []
for mn in motifNames:
    curClass = motifs[mn]['motifClass'] 
    classToMotifs[curClass].append(mn)

print("computing song features...")
for sn in songNames:
    songs[sn]['feats'] = getFeaturesForSongs(songs[sn]['score']);

# compute a bunch of feature vectors, add 'em to the motifs array
print("computing motif feature vectors...")
for mn in motifNames:
    motifs[mn]['feats'] = getFeatureVector(motifs[mn],songs);

#get featureVector keys in a roundabout way
featureVectorKeys = list(motifs[motifNames[0]]['feats'].keys())

#normalize featureVector entries
print("normalizing feature vectors...")
for fvk in featureVectorKeys:
    thisFeature = []
    
    #get list of all computed values for this feature
    for mn in motifNames:
        thisFeature.append(motifs[mn]['feats'][fvk])
    
    thisMean = numpy.mean(thisFeature)
    thisStd = numpy.std(thisFeature)
    
    for mn in motifNames:
        motifs[mn]['feats'][fvk] -= thisMean
        motifs[mn]['feats'][fvk] /= thisStd    

    
#split into a test set and a train+validate set. want to make sure that every
#motif class is represented in all sets, so we should snip sets off from the
#classToMotifs dictionary.

testMotifNames = []
trainAndValClassToMotifs = {}
trainAndValMotifNames = [] #a list of all non-test motifs, for testing

#first shuffle every list in the dict
for mc in motifClasses:
    random.shuffle(classToMotifs[mc])
    splitPos = int(numpy.floor(PROP_TESTING * len(classToMotifs[mc])))
    testMotifNames += classToMotifs[mc][:splitPos]
    trainAndValClassToMotifs[mc] = classToMotifs[mc][splitPos:]

#all features selected, for testing
defaultWeights = [1] * len(featureVectorKeys)

#function to pass to DEAP to test a single set of weights
def testWeights(wts,rotAmt):

    #using rotAmt, split the trainAndValClassToMotifs monstrosity into a
    #training set and a validation set, making sure again that every motif
    #class is represented in every set.
    trainMotifNames = []
    validateMotifNames = []
    
    for mc in motifClasses:
        rollAmt = int(len(trainAndValClassToMotifs[mc]) * rotAmt)
        numpy.roll(trainAndValClassToMotifs[mc],rollAmt)
        splitPos = int(PROP_VALIDATION * len(trainAndValClassToMotifs[mc]) / (1 - PROP_TESTING) )
        trainMotifNames += trainAndValClassToMotifs[mc][splitPos:]
        validateMotifNames += trainAndValClassToMotifs[mc][:splitPos]
    
    #pass the training and validation set to the performKNN method.
    res = performKNN(motifs,trainMotifNames,validateMotifNames,wts,K_NEAREST)
    print(str(round(res,5)) + " for " + str(wts))
    
    return (res,)

#function to make a random weight from {0, 1}
#RIGHT NOW ONLY FEATURE SELECTION WILL WORK! DON'T CHANGE THIS!
def instAttribute():
    var = [0,1]
    return random.choice(var)

numAttributes = len(featureVectorKeys)
#results = runGA(instAttribute,numAttributes,testWeights) 
    