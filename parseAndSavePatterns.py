# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 12:59:56 2018

@author: Tim
"""

import music21 as m21
import os
import csv
import numpy as np
import random
import featureExtractors as ft
import pickle
import copy
from collections import Counter
import patternClass as pc
from importlib import reload

reload(pc)
reload(ft)

### FILE SETUP, FETCH DATA
### -----------------------
thisFileDir = os.path.dirname(os.path.realpath('__file__'))
genPatternsfileDir = os.path.join(thisFileDir, '../genPatterns')
fileDir = os.path.join(thisFileDir, '../MTC-ANN-2.0.1/krn')
annFile = os.path.join(thisFileDir, '../MTC-ANN-2.0.1/metadata/MTC-ANN-motifs.csv')
tuneFamFile = os.path.join(thisFileDir, '../MTC-ANN-2.0.1/metadata/MTC-ANN-tune-family-labels.csv')

#get all parsed pattern files
genPatternFiles = []
for root, dirs, files in os.walk(genPatternsfileDir):
    for file in files:
        if file.endswith('.txt'):
            genPatternFiles.append(os.path.join(genPatternsfileDir,file))

#get all the song files into songNames, sort them
songNames = []
for root, dirs, files in os.walk(fileDir):
    for file in files:
        if file.endswith('.krn'):
            songNames.append(file[:-4])
songNames = sorted(songNames)

#make dictionary linking filenames -> songs of each score
print("fetching song scores...")
songs = {}
for i in range(0,len(songNames)):
    f = songNames[i]
    # songs[f] = {
    #         'score': m21.converter.parse(os.path.join(fileDir, f) + ".krn"),
    #         'songFeatures':None
    #         }
    songs[f] = pc.Song(
            score = m21.converter.parse(os.path.join(fileDir, f) + ".krn"),
            songFeatures = None
            )

print("fetching generated and annotated patterns...")
pClasses = {}
pOccs = {}

genPClassNames = [] #'generated patterns'
genPOccNames = []
genPTable = [];

annPClassNames = [] #'annotated patterns'
annPOccNames = []
annPTable = [];

#get entire generated pattern files and concatenate them into one big table

for gpf in genPatternFiles:
    with open(gpf, 'r') as fp:
        reader = csv.reader(fp, skipinitialspace=True,
                            delimiter=',', quotechar='"')
        curFileTable = [row for row in reader]
        genPTable = genPTable + curFileTable

#get entire annotated pattern file
with open(annFile, 'r') as fp:
    reader = csv.reader(fp, delimiter=',', quotechar='"')
    annPTable = [row for row in reader]

### EXTRACT PATTERN OCCURRENCES, SET UP OCCURRENCE TABLE
### --------------------------------------------------

#get unique set of pattern class names
genPClassNames = sorted(list(set([entr[0] for entr in genPTable])))
annPClassNames = sorted(list(set([entr[9] for entr in annPTable])))
allClassNames = annPClassNames + genPClassNames

#initialize dict structure for each class name. compute features later
for nm in (annPClassNames):
    pClasses[nm] = pc.PatClass(occNames=[],classFeatures={},type='ann');
for nm in (genPClassNames):
    pClasses[nm] = pc.PatClass(occNames=[],classFeatures={},type='gen');

#go thru pattern files and populate genpOccs and genpClasses
#THE HEADERS ARE:
#0: pattern class name
#1: pattern occurrence name
#2: song number (when songNames is sorted alphabetically)
#3: occurrence start index
#4: occurrence end index
for row in genPTable:
    #figure out new name for this pattern occurrence
    #row 0 is the pattern class name.

    thisOccPClass = row[0]
    occName = row[1]

    genPOccNames.append(occName)

    thisOccSongName = songNames[int(row[2])]
    thisOccStartInd = int(row[3])
    thisOccEndInd = int(row[4])

    thisOccScore = ft.extractPatternOccurrence(thisOccSongName,thisOccStartInd,
                                            thisOccEndInd,False,songs)
    pOccs[occName] = pc.PatOccurrence(
        songName=thisOccSongName,
        startInd=thisOccStartInd,
        endInd=thisOccEndInd,
        score=thisOccScore,
        patternClass=thisOccPClass,
        type='gen',
        occFeatures={} #compute features later
    )

    #add this occurrence's name to its corresponding pClasses entry
    pClasses[thisOccPClass].occNames.append(occName)

#do the same thing for our annotated patterns
#THE HEADERS ARE:
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
for row in annPTable:

    occName = row[2]
    annPOccNames.append(occName)

    thisOccSongName = row[1]
    thisOccPClass = row[9]
    thisOccStartInd = int(row[6])
    thisOccEndInd = int(row[7])
    thisOccScore = ft.extractPatternOccurrence(thisOccSongName,thisOccStartInd,
                                            thisOccEndInd,True,songs)
    pOccs[occName] = pc.PatOccurrence(
        songName = thisOccSongName,
        startInd = thisOccStartInd,
        endInd = thisOccEndInd,
        score = thisOccScore,
        patternClass = thisOccPClass,
        type = 'ann',
        occFeatures = {} #compute features later
    )

    #add this occurrence's name to its corresponding pClasses entry
    pClasses[thisOccPClass].occNames.append(occName)


###FILTER GENERATED PATTERN CLASSES
###---------------------------------------
print('filtering generated pattern classes...')
filtGenPClassNames = ft.filterPClassesWithKNN(annPClassNames,genPClassNames,1,
                                        pClasses,pOccs)
filtGenPOccNames = []
for gcn in filtGenPClassNames:
    filtGenPOccNames += pClasses[gcn].occNames

### COMPUTE FEATURE VECTORS
### -----------------------
#order of precedence = songs -> pattern occurrences -> pattern classes
print('computing features on songs...')
for sn in songNames:
    songs[sn].songFeatures = ft.getFeaturesForSongs(songs[sn].score);

print('computing features on occurrences...')
totalOccs = len(genPOccNames) + len(annPOccNames)
ct = 0

for on in (annPOccNames + genPOccNames):
    pOccs[on].occFeatures = ft.getFeaturesForOccurrences(pOccs[on],songs);
    ct += 1
    if ct % 500 == 0:
        print("   completed " + str(ct) + "/" +  str(totalOccs))

print('computing features on pattern classes...')
for cn in (annPClassNames + genPClassNames):
    pClasses[cn].classFeatures = ft.getFeaturesForClasses(pClasses[cn],pOccs,songs)

pClassFeatureKeys = pClasses[annPClassNames[0]].classFeatures.keys()
pClassFeatureKeys = sorted(pClassFeatureKeys)

#normalize featureVector entries
print("normalizing feature vectors...")
for fvk in pClassFeatureKeys:
    thisFeature = []

    #get list of all computed values for this feature
    for cn in (annPClassNames + genPClassNames):
        thisFeature.append(pClasses[cn].classFeatures[fvk])

    thisMean = np.mean(thisFeature)
    thisStd = np.std(thisFeature)

    for cn in (annPClassNames + genPClassNames):
        pClasses[cn].classFeatures[fvk] -= thisMean
        pClasses[cn].classFeatures[fvk] /= thisStd
        pClasses[cn].classFeatures[fvk] = np.clip(
                pClasses[cn].classFeatures[fvk],-3,3)

#things to pickle:
print("saving results to file...")
with open('parsed_patterns.pik', 'wb') as f:
  pickle.dump(copy.deepcopy((songs,
                             pClasses,
                             pOccs,
                             annPClassNames,
                             annPOccNames,
                             genPClassNames,
                             genPOccNames,
                             filtGenPClassNames
                             )), f, -1)
