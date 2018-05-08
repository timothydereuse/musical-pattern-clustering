#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 14:47:22 2018

@author: tdereuse
"""

import music21 as m21
from collections import Counter
import numpy
import matplotlib.pyplot as plt

def notes_and_occs(class_names, pClasses):
    pat_lengths = []
    pat_occs = []
    for pcn in class_names:
        pc = pClasses[pcn]
        theseNames = pc.occNames
        ml = 0
        for mn in theseNames:
            thisOcc = pOccs[mn]
            ml += len(thisOcc.score)
        ml /= len(theseNames)
        pat_lengths.append(ml)
        pat_occs.append(len(pc.occNames))
    return pat_lengths, pat_occs

x,y = notes_and_occs(annPClassNames,pClasses)
x2,y2 = notes_and_occs(filtGenPClassNames,pClasses)
plt.clf()
annpts = plt.scatter(x,y,color='black',marker='x',label='Annotated Patterns')
genpts = plt.scatter(x2,y2,color='black',marker='o',facecolors='none',label='Filtered Generated Patterns')
plt.ylabel("Number of Occurrences")
plt.xlabel("Mean Number of Notes per Occurence")
plt.legend(handles=[annpts,genpts])
axes = plt.gca()
axes.set_xlim([2,20.0])
axes.set_ylim([0,85])
#plt.savefig('testannPatClasses.png', format='png', figsize=(4,3), bbox_inches='tight')
plt.savefig('filteredplusannpatclasses.eps', format='eps', dpi=1000, bbox_inches='tight')

plt.clf()
x,y = notes_and_occs(genPClassNames,pClasses)
plt.scatter(x,y,color='black',marker='o')
plt.ylabel("Number of Occurrences")
plt.xlabel("Mean Number of Notes per Occurence")
axes = plt.gca()
axes.set_xlim([2,20.0])
axes.set_ylim([0,85])
#plt.savefig('genPatClasses.png', format='png', figsize=(4,3), bbox_inches='tight')
plt.savefig('allgenpatclasses.eps', format='eps', dpi=1000, bbox_inches='tight')

#indLengths = [ len(motifs[mn]['notes']) for mn in motifNames ]
