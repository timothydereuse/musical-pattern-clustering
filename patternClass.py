# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 15:42:50 2018

@author: Tim
"""


class Song(object):
    __slots__ = ['score', 'songFeatures', 'tuneFamily']

    def __init__(self, score, songFeatures, tuneFamily):
        self.score = score
        self.songFeatures = songFeatures
        self.tuneFamily = tuneFamily


class PatOccurrence(object):
    __slots__ = ['songName', 'startInd', 'endInd', 'score', 'patternClass', 'type', 'occFeatures', 'tuneFamily']

    def __init__(self, songName, startInd, endInd, score, patternClass, type, occFeatures, tuneFamily):
        self.songName = songName
        self.startInd = startInd
        self.endInd = endInd
        self.patternClass = patternClass
        self.score = score
        self.type = type
        self.occFeatures = occFeatures
        self.tuneFamily = tuneFamily


class PatClass(object):
    __slots__ = ['occNames', 'classFeatures', 'type', 'tuneFamily']

    def __init__(self, occNames, classFeatures, type, tuneFamily):
        self.occNames = occNames
        self.classFeatures = classFeatures
        self.type = type
        self.tuneFamily = tuneFamily
