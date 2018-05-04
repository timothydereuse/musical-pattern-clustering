# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 15:42:50 2018

@author: Tim
"""
class Song(object):
    __slots__ = ['score', 'songFeatures']

    def __init__ (self,score,songFeatures):
        self.score = score
        self.songFeatures = songFeatures

class PatOccurrence(object):
    __slots__ = ['songName','startInd','endInd',
                    'score','patternClass','type','occFeatures']

    def __init__ (self,songName,startInd,endInd,
                    score,patternClass,type,occFeatures):
        self.songName = songName
        self.startInd = startInd
        self.endInd = endInd
        self.patternClass = patternClass
        self.score = score
        self.type = type
        self.occFeatures = occFeatures

class PatClass(object):
    __slots__ = ['occNames','classFeatures','type']

    def __init__ (self,occNames,classFeatures,type):
        self.occNames = occNames
        self.classFeatures = classFeatures
        self.type = type
