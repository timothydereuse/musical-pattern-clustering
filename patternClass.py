# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 15:42:50 2018

@author: Tim
"""
from collections import namedtuple

Occurrence = namedtuple('Occurrence', ['songName','startInd','endInd',
                'score','patternClass','type','occFeatures'])

PatternClass = namedtuple('PatternClass', [])
