import os
import matplotlib.pyplot as plt
import pickle
import numpy as np

note_length_mult = 4

def get_note_bounds(pOccs):
    max_note = []
    min_note = []
    last_loc = []
    for occ_name in pOccs:
        notes = pOccs[occ_name].score.notes.stream()
        pitches = [x.pitch.midi for x in notes]
        max_note.append(max(pitches))
        min_note.append(min(pitches))

        locs = np.array([float(x.offset) for x in notes])
        locs = np.round((locs - min(locs)) * note_length_mult).astype('int')
        last_loc.append(max(locs))
    return max(max_note), min(min_note), max(last_loc)


def get_roll_from_class(p_class, pOccs, bounds):

    highest_note = bounds[0] + 1
    lowest_note = bounds[1]
    last_loc = bounds[2] + 1

    occs = p_class.occNames
    #setup pass
    all_notes = []
    for occ_name in occs:
        notes = pOccs[occ_name].score.notes.stream()
        pitches = [x.pitch.midi - lowest_note for x in notes]

        locs = np.array([float(x.offset) for x in notes])
        locs = np.round((locs - min(locs)) * note_length_mult).astype('int')

        all_notes.append(tuple(zip(locs, pitches)))

    all_notes_flat = [j for sub in all_notes for j in sub]
    # last_loc = max(x[0] for x in all_notes_flat)

    roll = np.zeros((last_loc, highest_note - lowest_note))

    # constuction pass
    for n in all_notes_flat:
        roll[n] += 1

    return roll


def assemble_rolls(normalize=True):
    print("loading data from file...")
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

    bounds = get_note_bounds(pOccs)

    labels = []
    data = []

    for i, cn in enumerate(pClasses.keys()):
        roll = get_roll_from_class(pClasses[cn], pOccs, bounds)

        if normalize:
            roll = np.array(roll) / max(roll.ravel())

        pClasses[cn].classFeatures = roll
        data.append(roll)
        labels.append(int(pClasses[cn].type == 'ann'))

    return np.array(data), np.array(labels)

def assemble_feats():
    print("loading data from file...")
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

    sorted_fkeys = sorted(list(pClasses.values())[0].classFeatures.keys())

    labels = []
    data = []
    for class_name in pClasses.keys():
        pClass = pClasses[class_name]
        feats = []
        for fkey in sorted_fkeys:
            feats.append(pClass.classFeatures[fkey])

        data.append(np.array(feats))
        labels.append(int(pClasses[class_name].type == 'ann'))

    return np.array(data), np.array(labels)

if __name__ == '__main__':
    data, labels = assemble_feats()
