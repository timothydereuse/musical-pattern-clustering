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
from timeit import default_timer as timer

motifLengths = []
motifOccs = []
for mc in motifClasses:
    theseNames = classToMotifs[mc]
    ml = 0
    for mn in theseNames:
        thisMotif = motifs[mn]
        ml += len(thisMotif['notes'])
    ml /= len(theseNames)
    motifLengths.append(ml)
    motifOccs.append(len(classToMotifs[mc]))
    
plt.scatter(motifLengths,motifOccs,color='black')
plt.ylabel("Number of Occurrences")
plt.xlabel("Mean Number of Notes per Occurence")
plt.savefig('dbgraph1.eps', format='eps', dpi=1000, bbox_inches='tight')
    
#indLengths = [ len(motifs[mn]['notes']) for mn in motifNames ]
