import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
import itertools

note_length_mult = 4
pickle_name = 'parsed_patterns.pik'


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
    # setup pass
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


def dict_to_array(feat_dict, sorted_keys):
    x = [feat_dict[fkey] for fkey in sorted_keys]
    return x


def assemble_rolls(normalize=True):
    print("loading data from file...")
    with open(pickle_name, "rb") as f:
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
    with open(pickle_name, "rb") as f:
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


def assemble_clustering_feats(max_similar=0, unsimilar_factor=0.1, gen_factor=3):
    print("loading data from file...")
    with open(pickle_name, "rb") as f:
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

    similar_pairs = []
    unsimilar_pairs = []

    # get all 2-combinations of each class
    for class_name in annPClassNames:
        occ_names = pClasses[class_name].occNames
        combo = list(itertools.combinations(occ_names, 2))
        similar_pairs += combo

        # choose occs from other classes
        other_occs = [x for x in annPOccNames if not (x in occ_names)]
        choose_other_occs = np.random.choice(other_occs, int(len(combo) * unsimilar_factor))
        choose_gen_occs = np.random.choice(genPOccNames, int(len(combo) * gen_factor), replace=False)

        for i, occ in enumerate(np.concatenate((choose_other_occs, choose_gen_occs))):
            this_class_occ = occ_names[i % len(occ_names)]
            unsimilar_pairs.append((this_class_occ, occ))

    if max_similar > 0:
        idxs1 = np.random.choice(range(len(similar_pairs)), max_similar)
        idxs2 = np.random.choice(range(len(unsimilar_pairs)), max_similar)
        similar_pairs = [similar_pairs[x] for x in idxs1]
        unsimilar_pairs = [unsimilar_pairs[x] for x in idxs2]

    data = []
    labels = []

    for pair in similar_pairs:
        feats1 = dict_to_array(pOccs[pair[0]].occFeatures, sorted_fkeys)
        feats2 = dict_to_array(pOccs[pair[1]].occFeatures, sorted_fkeys)
        asdf = np.array([feats1, feats2])
        data.append(asdf)
        labels.append(1)

    for pair in unsimilar_pairs:
        feats1 = dict_to_array(pOccs[pair[0]].occFeatures, sorted_fkeys)
        feats2 = dict_to_array(pOccs[pair[1]].occFeatures, sorted_fkeys)
        asdf = np.array([feats1, feats2])
        data.append(asdf)
        labels.append(-1)

    # sorted_fkeys = sorted(list(pClasses.values())[0].classFeatures.keys())
    #
    # labels = []
    # data = []
    # for class_name in pClasses.keys():
    #     pClass = pClasses[class_name]
    #     feats = []
    #     for fkey in sorted_fkeys:
    #         feats.append(pClass.classFeatures[fkey])
    #
    #     data.append(np.array(feats))
    #     labels.append(int(pClasses[class_name].type == 'ann'))

    return np.array(data), np.array(labels)


if __name__ == '__main__':
    # data, labels = assemble_feats()
    data, labels = assemble_clustering_feats()
