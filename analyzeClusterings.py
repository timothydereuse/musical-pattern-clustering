from music21 import *
import pickle
import matplotlib.pyplot as plt
import numpy as np


def reassemble_patterns(labelling, test_occs, pOccs):
    found_patterns = []
    occ_name_to_pattern = {}
    for pat_num in list(set(labelling)):
        idxs = [i for i, x in enumerate(labelling) if x == pat_num]
        occ_names = [test_occs[i] for i in idxs]
        occs = [pOccs[n] for n in occ_names]
        found_patterns.append(occ_names)
        for on in occ_names:
            occ_name_to_pattern[on] = pat_num
    return found_patterns, occ_name_to_pattern


def list_notes(occ_name, pOccs):
    occ = pOccs[occ_name]
    noteDurs = [round(float(x.quarterLength), 5) for x in occ.score.notes.stream()]
    noteNums = [x.name for x in occ.score.notes.stream()]
    return list(zip(noteNums, noteDurs)), occ.songName, occ.tuneFamily, occ.startInd


with open('parsed_patterns.pik', "rb") as f:
    dat = pickle.load(f)
songs = dat[0]
pClasses = dat[1]
pOccs = dat[2]
annPClassNames = dat[3]
annPOccNames = dat[4]
genPClassNames = dat[5]
genPOccNames = dat[6]
filtGenPClassNames = dat[7]
sorted_fkeys = sorted(list(pOccs.values())[0].occFeatures.keys())

with open('paperfigs.pik', 'rb') as f:
    asdf = pickle.load(f)
emb_labellings, labels_true, test_occs, dummy = asdf

emb_pats, emb_guide = reassemble_patterns(emb_labellings[3], test_occs, pOccs)
real_pats, real_guide = reassemble_patterns(labels_true, test_occs, pOccs)

EXAMPLE_FIG1 = [emb_pats[1][x] for x in [2, 3, 6, 10]]
EXAMPLE_FIG2A = [real_pats[17][x] for x in [1, 8, 18]]
EXAMPLE_FIG2B = [real_pats[17][x] for x in [13, 15, 11]]
EXAMPLE_FIG3 = emb_pats[18]

print('FIG 1')
for i, occ_name in enumerate(EXAMPLE_FIG1):
    notes = str(i) + ' ' + str(list_notes(occ_name, pOccs))
    notes += ' {}, {}'.format(real_guide[occ_name], emb_guide[occ_name])
    print(notes)

print('FIG 2a')
for i, occ_name in enumerate(EXAMPLE_FIG2A):
    notes = str(i) + ' ' + str(list_notes(occ_name, pOccs))
    notes += ' {}, {}'.format(real_guide[occ_name], emb_guide[occ_name])
    print(notes)

print('FIG 2b')
for i, occ_name in enumerate(EXAMPLE_FIG2B):
    notes = str(i) + ' ' + str(list_notes(occ_name, pOccs))
    notes += ' {}, {}'.format(real_guide[occ_name], emb_guide[occ_name])
    print(notes)

print('FIG 3')
for i, occ_name in enumerate(EXAMPLE_FIG3):
    notes = str(i) + ' ' + str(list_notes(occ_name, pOccs))
    notes += ' {}, {}'.format(real_guide[occ_name], emb_guide[occ_name])
    print(notes)

# with open('paperfigs.pik', 'wb') as f:
#     pickle.dump((emb_labellings, labels_true, test_occs, "labelling 3 was the one used for figures"), f, -1)
