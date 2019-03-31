import torch
import pickle
import clusteringTest as ct
import distanceLearningNet as dln
import prepareDataForTraining as pdft
import netClasses as nc
from importlib import reload
reload(ct)
reload(dln)
reload(pdft)

import numpy as np

num_validation_sets = 8 # number of experiments to run
val_ratio = 0.1         # use this much of each training set for validation


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
pickle_name = 'parsed_patterns.pik'

# load from pickle
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

# break up into sets
idx_shuffle = np.array(range(len(annPClassNames)))
np.random.shuffle(idx_shuffle)
set_idxs = np.array_split(idx_shuffle, num_validation_sets)

all_results = []

for run_num in range(num_validation_sets):
    print("starting run {}...".format(run_num))

    test_idxs = set_idxs[0]
    train_idxs = np.concatenate(set_idxs[1:])
    set_idxs = np.roll(set_idxs, 1)    # prepare for the next test by rotating test/train

    # get pairwise similarity/unsimilarity data for training set
    train_class_names = [annPClassNames[x] for x in train_idxs]
    test_class_names = [annPClassNames[x] for x in test_idxs]

    train_data, train_labels = pdft.assemble_clustering_feats(dat,
        train_class_names,
        unsimilar_factor=4,
        gen_factor=4,
        max_similar=0
        )

    # make the model
    model = nc.FFNetDistance(num_feats=train_data.shape[-1])
    model.to(device)

    val_split_idx = int(len(train_data) * val_ratio)
    pair_idx_shuffle = np.array(range(len(train_data)))
    np.random.shuffle(pair_idx_shuffle)
    val_pair_idxs = pair_idx_shuffle[:val_split_idx]
    train_pair_idxs = pair_idx_shuffle[val_split_idx:]

    x_train = torch.tensor(train_data[train_pair_idxs]).float()
    y_train = torch.tensor(train_labels[train_pair_idxs]).long()
    x_val = torch.tensor(train_data[val_pair_idxs]).float()
    y_val = torch.tensor(train_labels[val_pair_idxs]).long()

    model, accs = dln.train_model((x_train, y_train), model, device,
        batch_size=128,
        num_epochs=50000,
        stagnation_time=1000,
        poll_every=1000,
        val_every=50,
        lr=2e-4,
        val_data=(x_val, y_val)
        )

    # TESTING
    # assemble test occurrences
    model.eval() # set model to evaluation mode

    test_occs = []
    labels_true = []
    for i, pn in enumerate(test_class_names):
        occNames = pClasses[pn].occNames
        for on in occNames:
            test_occs.append(on)
            labels_true.append(i)

    # add noisy occs:
    for i in range(1):
        test_occs.append(str(np.random.choice(genPOccNames)))
        labels_true.append(-1)

    res = ct.evaluate_clustering(test_occs, labels_true, model, pOccs)
    print(res)
    all_results.append(res)

for key in res.keys():
    category = [x[key] for x in all_results]
    mean = np.mean(category)
    stdv = np.std(category)
    print(key, mean, stdv)
