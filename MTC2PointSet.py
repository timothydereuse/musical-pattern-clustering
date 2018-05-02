# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 16:04:16 2018

@author: Tim
"""

import music21 as m21
#m21.environment.set('musescoreDirectPNGPath', 'C:\\Program Files (x86)\\MuseScore 2\\bin\\MuseScore.exe')
import os
#import numpy
import csv
from fractions import Fraction

#and here we attempt to gather up all the 360 songs in MTC-ANN and output them
#into a concatenated point-set representation, using music21
#we'll start with maybe just a single tune family, though.

thisfileDir = os.path.dirname(os.path.realpath('__file__'))
fileDir = os.path.join(thisfileDir, '../MTC-ANN-2.0.1/krn')
annFile = os.path.join(thisfileDir, '../MTC-ANN-2.0.1/metadata/MTC-ANN-motifs.csv')
tuneFamFile = os.path.join(thisfileDir, '../MTC-ANN-2.0.1/metadata/MTC-ANN-tune-family-labels.csv')

#for now points will just be (start time, midi pitch, duration)

#get all the files into songNames
songNames = []

for root, dirs, files in os.walk(fileDir):

    for file in files:
        if file.endswith('.krn'):
            songNames.append(file[:-4])
    
#make dictionary linking filenames -> songs of each score
print("fetching scores...")
songs = {}

#get songnames in sorted order!!
songNames = sorted(songNames)

for i in range(0,len(songNames)):
    f = songNames[i]
    songs[f] = {'score': m21.converter.parse(os.path.join(fileDir, f) + ".krn")}
    #curScore = songs[f]['score'].flat.notes.stream()
    curScore = songs[f]['score'].flat.notes.stream()   
    pointSet = []
    
    for n in curScore.iter.notes:
        off = n.offset
        midiVal = n.pitch.midi
        dur = n.quarterLength
        
        #this is incredibly stupid but i'm not sure how else to keep
        #each song totally separate
        pointSet.append((off,midiVal,dur,i))
        
    songs[f]['pointSet'] = pointSet

#dictionary linking tune family names -> lists of song names that belong to 
#each tune family
tuneFamToSongNames = {};

with open(tuneFamFile) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if row[1] in tuneFamToSongNames:
            tuneFamToSongNames[row[1]].append(row[0])
        else:
            tuneFamToSongNames[row[1]] = [row[0]]

print("writing to file...")

    
for tuneFamName in tuneFamToSongNames.keys():
    
    fname = "tuneFam_" + tuneFamName[0:12] + ".txt"
    file = open(fname,"w")
    tuneFamList = sorted(tuneFamToSongNames[tuneFamName])
    
    #to be compatible with the examples in PattDisc we want no decimals;
    #all fractions
    for i in range(0,len(tuneFamList)):
        sn = tuneFamList[i]
        for pt in songs[sn]['pointSet']:
            file.write("(") 
            for j in range(0,len(pt)):
                if j != 0:
                    file.write(" ")
                
                if type(pt[j]) == Fraction:
                    file.write(str(pt[j].numerator) + "/" + str(pt[j].denominator))
                elif (pt[j] != int(pt[j])):
                    temp = Fraction(pt[j]).limit_denominator(10)
                    file.write(str(temp.numerator) + "/" + str(temp.denominator))
                else:
                    file.write(str(int(pt[j])))
            file.write(")\n")
    
    file.close() 

    







