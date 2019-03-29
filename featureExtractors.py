# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 17:42:18 2018

@author: Tim
"""
import music21 as m21
import music21.features.jSymbolic as jsym
import scipy.stats
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

#round all duration values to this many digits!
#some are stored as fractions and that's just inconvenient
ROUND_DURS_DIGITS = 5;

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

#try to fetch a single motif


# def extractMotif(annEntry, songs):
#     """
#     given a row from the annotation file and the database of score files,
#     return the notes of theassociated motif and some of its metadata as a
#     dictionary.
#     """
#
#     songName = annEntry[1]
#     inStart = int(annEntry[6])
#     numNotes = int(annEntry[8])
#
#     #add number of ties before start index from start index; meertens
#     #DOESN'T count tied notes as notes but music21 DOES
#     allNotes = songs[songName].score.flat.notes.stream()
#     #subtract 1 here to get the first note of the occurence in the slice
#     #so that we can get rid of it if it's a rest
#     beforeSlice = allNotes[:inStart-1]
#     numTies = 0
#     for n in beforeSlice:
#         if(n.tie != None):
#             if(n.tie.type == 'start'):
#                 numTies += 1
#
#     inStart += numTies
#
#     #do the same for ties inside of the snippet, but also keep track of where
#     #they are and save that information with the motif so we don't have to go
#     #through this procedure again
#     numTies = 0
#     inSlice = allNotes[inStart:(inStart+numNotes)]
#     for n in inSlice:
#         if(n.tie != None):
#             if(n.tie.type == 'start'):
#                 numTies += 1
#
#
#     #this new numNotes will work with music21
#     numNotes += numTies
#
#     #NOW we know that we have the actual motif!
#     motif = allNotes[inStart:(inStart+numNotes)]
#
#     return {'notes':motif,
#             'startInd':inStart,
#             'endInd':(inStart+numNotes),
#             'songID':annEntry[1],
#             'motifClass':annEntry[9],
#             'duration':annEntry[5]}

#annotated first starting at 0, but tied notes are only counted for the onset
#must disregard tied notes when doing start/end indices tabarnak

#so: consider the list of notes up to the first index. if there's n ties
#that live behind the start index, increment the start index by n. when done,
#look 8 notes ahead and do the same thing
def extractPatternOccurrence(songName,inStart,inEnd,useTies,songs):
    """
    given song name, occurrence start, occurrence end, and the database of score files,
    return the notes of the associated pattern occurrence
    useTies is a boolean determining whether or not tied notes count as
    two notes or one for the purpose of indexing (true for 1, false for 2)
    necessary bc MTC-ANN indexing doesn't count
    """

    #inStart = int(annEntry[6])
    #numNotes = int(annEntry[8])
    numNotes = inEnd - inStart + 1 #including endpoints

    #add number of ties before start index from start index; meertens
    #DOESN'T count tied notes as notes but music21 DOES
    allNotes = songs[songName].score.flat.notes.stream()
    #subtract 1 here to get the first note of the occurence in the slice
    #so that we can get rid of it if it's a rest
    if(useTies):
        beforeSlice = allNotes[:inStart-1]
        numTies = 0
        for n in beforeSlice:
            if(n.tie != None):
                if(n.tie.type == 'start'):
                    numTies += 1

        inStart += numTies

        #do the same for ties inside of the snippet, but also keep track of where
        #they are and save that information with the pattOcc so we don't have to go
        #through this procedure again (TODO)
        numTies = 0
        inSlice = allNotes[inStart:(inStart+numNotes)]
        for n in inSlice:
            if(n.tie != None):
                if(n.tie.type == 'start'):
                    numTies += 1

        #this new numNotes will work with music21
        numNotes += numTies

    pattOcc = allNotes[inStart:(inStart+numNotes)]

    return pattOcc

def getFeaturesForSongs(score):
    vec = {}

    mel = score.flat.notes.stream()
    noteNums = [x.pitch.midi for x in mel]
    intervals = [noteNums[n] - noteNums[n-1] for n in range(1,len(noteNums))]
    couInt = dict(Counter(intervals))
    for k in couInt.keys():
        couInt[k] /= len(intervals)

    vec['interval_probs'] = couInt
    vec['pitch_mean'] = np.mean(noteNums)
    vec['interval_mean'] = np.mean(np.abs(intervals))
    vec['interval_signs'] = sum(np.sign(intervals)) / len(intervals)
    vec['interval_prop_small'] = sum([abs(intervals[n]) <= 2 for n in range(0,len(intervals))]) / len(intervals)
    vec['interval_prop_large'] = sum([abs(intervals[n]) >= 7 for n in range(0,len(intervals))]) / len(intervals)

    noteDurs = [round(float(x.quarterLength),ROUND_DURS_DIGITS) for x in mel]

    couRtm = dict(Counter(noteDurs))
    for k in couRtm.keys():
        couRtm[k] /= len(noteDurs)

    vec['duration_probs'] = couRtm
    vec['rhythm_density'] = np.mean(noteDurs)
    vec['rhythm_variability'] = np.std([np.log(float(n)) for n in noteDurs]) #from Collins 2014

    # HISTOGRAMS:
    # interval counting
    for n in range(13):
        num = len([x for x in intervals if abs(x) == n])
        vec['interval_count_' + str(n)] = num / len(intervals)
    for n in range(12):
        num = len([x for x in noteNums if abs(x) % 12 == n])
        vec['pitch_class_count_' + str(n)] = num / len(noteNums)
    for n in range(-3,3):
        num = len([x for x in noteDurs if 2**(n) <= x < 2**(n+1)])
        vec['rhythm_duration_count_' + str(n)] = num / len(noteDurs)

    return vec

#single method that is passed an entry from the motifs dict
#and the database of songs and returns a dict that is a feature
#vector for that motif. here go
def getFeaturesForOccurrences(cur_class,songs):

    #DERIVATIVES!!!
    #first do as much as we can with just the notes belonging
    #to the motif itself and see how that does
    vec = {}
    mel = cur_class.score

    #for now just remove rests

    noteNums = [x.pitch.midi for x in mel]
    intervals = [noteNums[n] - noteNums[n-1] for n in range(1,len(noteNums))]

    highest = max(noteNums)
    lowest = min(noteNums)

    vec['numNotes'] = len(noteNums)

    vec['pitch_highest'] = highest
    vec['pitch_lowest'] = lowest
    vec['pitch_range'] = highest-lowest
    vec['pitch_num_classes'] = len(set(noteNums))
    vec['pitch_mean'] = np.mean(noteNums)
    vec['pitch_std'] = np.std(noteNums)
    vec['pitch_pos_highest'] = noteNums.index(highest) / len(noteNums)
    vec['pitch_pos_lowest'] = noteNums.index(lowest) / len(noteNums)

    # pitch counting
    for n in range(12):
        num = len([x for x in noteNums if abs(x) % 12 == n])
        vec['pitch_class_count_' + str(n)] = num / len(noteNums)

    vec['interval_max'] = max(np.abs(intervals))
    vec['interval_min'] = min(np.abs(intervals))
    vec['interval_largest_asc'] = max([max(intervals),0])
    vec['interval_largest_desc'] = min([min(intervals),0])
    vec['interval_mean'] = np.mean(np.abs(intervals))
    vec['interval_prop_small'] = sum([abs(intervals[n]) <= 2 for n in range(0,len(intervals))]) / len(intervals)
    vec['interval_prop_large'] = sum([abs(intervals[n]) >= 7 for n in range(0,len(intervals))]) / len(intervals)
    vec['interval_asc_or_desc'] = np.sign(noteNums[0] - noteNums[len(noteNums)-1])
    vec['interval_signs'] = sum(np.sign(intervals)) / len(intervals)

    # interval counting
    for n in range(13):
        num = len([x for x in intervals if abs(x) == n])
        vec['interval_count_' + str(n)] = num / len(intervals)

    #-1 if monotonically down, 1 if up, else 0
    if all([np.sign(x) == 1 for x in intervals]):
        vec['interval_strict_asc_or_desc'] = 1
    elif all([np.sign(x) == -1 for x in intervals]):
        vec['interval_strict_asc_or_desc'] = -1
    else:
        vec['interval_strict_asc_or_desc'] = 0

    #rhythmic properties
    noteDurs = [round(float(x.quarterLength),ROUND_DURS_DIGITS) for x in mel]
    vec['rhythm_duration'] = sum(noteDurs)
    vec['rhythm_longest_note'] = max(noteDurs)
    vec['rhythm_shortest_note'] = min(noteDurs)
    vec['rhythm_density'] = np.mean(noteDurs)
    vec['rhythm_variability'] = np.std([np.log(float(n)) for n in noteDurs]) #from Collins 2014
    vec['rhythm_last_note_duration'] = noteDurs[len(noteDurs)-1]

    # rhythm counting
    for n in range(-3,3):
        num = len([x for x in noteDurs if 2**(n) <= x < 2**(n+1)])
        vec['rhythm_duration_count_' + str(n)] = num / len(noteDurs)

    #POLYFIT IDEA
    yCoords = [y - noteNums[0] for y in noteNums]
    xtemp = [float(x.offset) / vec['rhythm_duration'] for x in mel]
    xCoords = [x - xtemp[0] for x in xtemp]

    #print(str(xCoords) + " vs " + str(yCoords))
    polyFit1 = np.polyfit(xCoords,yCoords,1,full=True)
    vec['polyfit_1'] = polyFit1[0][0]
    vec['polyfit_residual_1'] = 0
    if polyFit1[1].size > 0:
        vec['polyfit_residual_1'] = np.sqrt(polyFit1[1][0])

    vec['polyfit_2'] = 0
    vec['polyfit_residual_2'] = 0
    vec['polyfit_3'] = 0
    vec['polyfit_residual_3'] = 0

    if len(noteNums) >= 3:
        polyFit2 = np.polyfit(xCoords,yCoords,2,full=True)
        vec['polyfit_2'] = polyFit2[0][0]
        if polyFit2[1].size > 0:
            vec['polyfit_residual_2'] = np.sqrt(polyFit2[1][0])

    if len(noteNums) >= 4:
        polyFit3 = np.polyfit(xCoords,yCoords,3,full=True)
        vec['polyfit_3'] = polyFit3[0][0]
        if polyFit3[1].size > 0:
            vec['polyfit_residual_3'] = np.sqrt(polyFit3[1][0])

    #differences between song and this motif
    songVec = songs[cur_class.songName].songFeatures

    song_diff_keys = [
            'interval_mean',
            'rhythm_variability',
            'rhythm_density',
            'interval_signs',
            'pitch_mean',
            'interval_prop_small',
            'interval_prop_large'
            ]
    song_diff_keys += [x for x in vec.keys() if '_count' in x]

    for key in song_diff_keys:
        vec['diff_' + key] = songVec[key] - vec[key]

     #songScore = songs[motif['songName']]['score'].flat.notes.stream()
#    songScoreNums = [x.pitch.midi for x in songScore]

#    vec['intervalFollowing'] = 0
#    if motif['endInd'] + 1 < len(songScoreNums):
#        vec['intervalFollowing'] = songScoreNums[motif['endInd'] + 1] - noteNums[-1]
#    vec['intervalPreceding'] = 0
#    if motif['endInd'] - 1 > 0:
#        vec['intervalPreceding'] = songScoreNums[motif['endInd'] - 1] - noteNums[0]


    sumIntProbs = 1
    for i in intervals:
        sumIntProbs *= songVec['interval_probs'][i]
    vec['interval_log_expected_occurrences'] = np.log(sumIntProbs)

    sumDurProbs = 1
    for d in noteDurs:
        sumDurProbs *= songVec['duration_probs'][d]
    vec['rhythm_log_expected_occurrences'] = np.log(sumDurProbs)

    vec['rhythm_starts_on_downbeat'] = 0
    vec['rhythm_crosses_measure'] = 0
    vec['rhythm_start_beat_str'] = 0
    vec['rhythm_last_beat_str'] = 0
    try:
        noteBeats = [x.beat for x in mel]
        vec['rhythm_starts_on_downbeat'] = (noteBeats[0] == 1.0)
        vec['rhythm_crosses_measure'] = sum([noteBeats[n] < noteBeats[n-1] for n in range(1,len(noteBeats))]) > 0

        #figure out how to tell if note has associated time signature
        noteStr = [x.beatStrength for x in mel]
        vec['rhythm_start_beat_str'] = np.log(noteStr[0])
        vec['rhythm_last_beat_str'] = np.log(noteStr[len(noteStr)-1])
    except m21.Music21ObjectException:
        #this is not a good solution.
        pass

    #send it back
    return vec


def getFeaturesForClasses(patternClass, occs, songs):
    #take the average/std over all occurrences
    vec = {}

    vec['numOccs'] = len(patternClass.occNames)

    occFeatureKeys = occs[patternClass.occNames[0]].occFeatures.keys()

    for fk in occFeatureKeys:
        allOccVals = [occs[occName].occFeatures[fk] for occName in patternClass.occNames]
        vec["avg_" + fk] = np.mean(allOccVals)
        vec["std_" + fk] = np.std(allOccVals)


    scores = [ occs[oc].score.flat for oc in patternClass.occNames]

    noteNums = [[x.pitch.midi for x in mel] for mel in scores]
    noteDurs = [[round(float(x.quarterLength),ROUND_DURS_DIGITS) \
    for x in mel] for mel in scores]

    flatNums = [x for subList in noteNums for x in subList]
    vec['num_notes_total'] = len(flatNums)

    vec['unique_pitch_prop_content'] = \
    len(set(tuple(x) for x in noteNums)) / vec['numOccs']

    vec['unique_rhythm_prop_content'] = \
   len(set(tuple(x) for x in noteDurs)) / vec['numOccs']

    pitchAndDurs = [(noteNums[x] + noteDurs[x]) for x in range(0,vec['numOccs'])]

    vec['prop_unique_content'] = \
    len(set(tuple(x) for x in pitchAndDurs)) / vec['numOccs']

    return vec


def filterPClassesWithKNN(annPClassNames, genPClassNames, kNearest, pClasses, pOccs):
    #so: we want to take a sample of our huge number of generated pattern classes
    #such that the number of occurrences and average cardinality doesn't easily
    #distinguish our sample from the annotated group.

    #perform a quick and dirty knn to get a bunch of generated class names
    #whose cardinalities and numOccs somewhat match the annotated data.
    indexPairs = np.arange(len(annPClassNames))
    indexPairs = np.concatenate([indexPairs, indexPairs])
    np.random.shuffle(indexPairs)
    indexPairs = np.split(indexPairs,len(indexPairs)/2)

    #deep copy!
    genPClassNamesCopy = list(genPClassNames)
    filtGenPClassNames = []

    for i in range(len(annPClassNames)):

        tar1 = pClasses[annPClassNames[indexPairs[i][0]]]
        tar2 = pClasses[annPClassNames[indexPairs[i][1]]]

        tarNumOccs = len(tar1.occNames)
        tar2Notes = [len(pOccs[on].score) for on in tar2.occNames]
        tarNumNotes = np.mean(tar2Notes)

        candidateNameList = []

        #calculate how close each generated class is to these parameters
        for gcn in genPClassNamesCopy:
            cand = pClasses[gcn]
            candNumOccs = len(cand.occNames)
            candNotes = [len(pOccs[on].score) for on in cand.occNames]
            candNumNotes = np.mean(candNotes)

            candScore = (candNumOccs - tarNumOccs)**2 + (candNumNotes - tarNumNotes)**2

            candidateNameList.append([candScore, gcn])

        #from the kNearest closest generated classes, choose one and remove
        #that one from the copy array
        candidateNameList = sorted(candidateNameList,key=lambda x: x[0])
        chop = candidateNameList[0:kNearest]
        choice = chop[np.random.choice(kNearest)][1]
        filtGenPClassNames.append(choice)
        genPClassNamesCopy.remove(choice)

    return filtGenPClassNames

def keys_subset(all_keys,type_string):
    if type_string == 'only_pitch':
        return [x for x in all_keys if ('pitch' in x or 'interval' in x)]
    elif type_string == 'only_rhythm':
        return [x for x in all_keys if ('rhythm' in x)]
    elif type_string == 'exclude_means':
        return [x for x in all_keys if ('avg' not in x)]
    elif type_string == 'exclude_stds':
        return [x for x in all_keys if ('std' not in x)]
    elif type_string == 'exclude_song_comp':
        return [x for x in all_keys if ('diff' not in x and 'expected' not in x)]
    elif type_string == 'all':
        return all_keys
    else:
        raise TypeError('bad keys_subset type ' + str(type_string))
    pass

def split_into_chunks(inp,num_chunks):

    chunk_len = int(np.floor(len(inp) / num_chunks))
    chunks = [inp[i:i + chunk_len] for i in range(0, len(inp), chunk_len)]
    if len(chunks) > num_chunks:
        for i,x in enumerate(chunks[num_chunks]):
            chunks[i].append(x)
        del chunks[num_chunks]

    return chunks

#just for testing: get all features
#plt.plot(sorted(inspectFeature('classAvg_pitch_mean',pClasses,genPClassNames + annPClassNames)))
def inspectFeature(featureName,table,tableNames,featsType="classFeatures"):
    ret = []
    for tn in tableNames:
        item = table[tn]
        ret.append(item[featsType][featureName])
    return ret

def scatterFeatures(fn1,fn2,table,tableNames):

    xs = []
    ys = []
    types = []

    for tn in tableNames:
        item = table[tn]
        xs.append(item.classFeatures[fn1])
        ys.append(item.classFeatures[fn2])
        if item['type'] == 'ann':
            types.append('r')
        else:
            types.append('k')

    print(types)

    plt.scatter(xs,ys,c=types)
    plt.xlabel(fn1)
    plt.ylabel(fn2)
    plt.show()
    return
