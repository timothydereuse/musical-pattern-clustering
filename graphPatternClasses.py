#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 14:47:22 2018

@author: tdereuse
"""

import music21 as m21
from collections import Counter
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
# Use LaTeX for rendering
matplotlib.rcParams["text.usetex"] = True
# load the xfrac package
matplotlib.rcParams["text.latex.preamble"].append(r'\usepackage[dvips]{graphicx}\usepackage{xfrac}')


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

plt.clf()
x,y = notes_and_occs(annPClassNames,pClasses)
x2,y2 = notes_and_occs(filtGenPClassNames,pClasses)
plt.clf()
plt.figure(figsize=(5,3))
annpts = plt.scatter(x,y,color='black',marker='x',s=20,label='Annotated Patterns')
genpts = plt.scatter(x2,y2,color='black',marker='o',s=20,facecolors='none',label='Filtered Generated Patterns')
plt.ylabel("Number of Occurrences")
plt.xlabel("Mean Number of Notes per Occurence")
plt.legend(handles=[annpts,genpts])
axes = plt.gca()
axes.set_xlim([2,20.0])
axes.set_ylim([0,85])
#plt.savefig('testimgannPatClasses.png', format='png', bbox_inches='tight')
plt.savefig('filteredplusannpatclasses.eps', format='eps', dpi=1000, bbox_inches='tight')

plt.clf()
plt.figure(figsize=(5,3))
x,y = notes_and_occs(genPClassNames,pClasses)
genpts = plt.scatter(x,y,color='black',marker='o',s=3,label='Generated Patterns')
plt.ylabel("Number of Occurrences")
plt.xlabel("Mean Number of Notes per Occurence")
plt.legend(handles=[genpts])
axes = plt.gca()
axes.set_xlim([2,20.0])
axes.set_ylim([0,85])
#plt.savefig('testimggenPatClasses.png', format='png', bbox_inches='tight')
plt.savefig('allgenpatclasses.eps', format='eps', dpi=1000, bbox_inches='tight')

#indLengths = [ len(motifs[mn]['notes']) for mn in motifNames ]

# HERE we compute pearson coefficients
def pearsonCoefficients(featureKeys, classNames, pClasses):
    results = []
    hist_results = []
    for k in featureKeys:

        featVals = []
        types = []

        for cn in (classNames):
            featVals.append(pClasses[cn].classFeatures[k])
            types.append(pClasses[cn].type == 'ann')

        res = scipy.stats.pearsonr(featVals,types)
        results.append((res[0],res[1],k))
        hist_results.append(res[0])

    res = ""
    results = sorted(results,key=lambda x: -1*np.abs(x[0]))

    for r in results:
        temp = "\\texttt{" + r[2] + "}"
        temp = temp.replace("_","\_")
        res += "{:<50}  & {:1.3} \\\\ \n".format(temp,r[0])

    plt.clf()
    plt.figure(figsize=(5,3))
    plt.hist(hist_results,bins = 15,color='0.5',ec='k',histtype='stepfilled')
    plt.ylabel("Number of Features")
    plt.xlabel("Pearson's Correlation Coefficient $\mathit{r}$")
    #plt.savefig('testimghist.png', format='png', figsize=(4,3), bbox_inches='tight')
    plt.savefig('pearsoncoeffhist.eps', format='eps', dpi=1000, bbox_inches='tight')

    return res

def test_plot():

    plt.figure(figsize=(5,3))
    x = np.linspace(0, 10, 1000)
    plt.plot(x, x,color='black');

    plt.ylabel("Y")
    plt.xlabel("X")

    plt.savefig('testplot.eps', format='eps', dpi=1000, bbox_inches='tight')
    #plt.savefig('testplot.png', format='png', figsize=(4,3), bbox_inches='tight')

    return 0
