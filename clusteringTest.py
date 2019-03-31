import os
import pickle
import numpy as np
import itertools
import torch
import netClasses as nc
from sklearn.cluster import DBSCAN
from sklearn import metrics

pickle_name = 'parsed_patterns.pik'

def evaluate_clustering(test_occs, labels_true, model, pOccs):
    sorted_fkeys = sorted(list(pOccs.values())[0].occFeatures.keys())

    def occ_to_subspace(occ):
        if type(occ) == str:
            occ = pOccs[occ]
            arr = [occ.occFeatures[fkey] for fkey in sorted_fkeys]
            arr = torch.tensor(arr).float()
        else:
            arr = occ
        sub = model.subspace(arr)
        return sub.detach().numpy()

    test_data = []
    # build feature vectors. careful with indices...
    with torch.no_grad():
        for occ_name in test_occs:
            test_data.append(occ_to_subspace(occ_name))
    test_data = np.array(test_data)

    eps_to_try = np.geomspace(1e-6, 1e-2, 100)
    best_db = None
    best_ep = 0
    best_sil = -1
    for ep in eps_to_try:
        db = DBSCAN(eps=0.025, metric='l1', min_samples=3).fit(test_data)
        # hom = metrics.homogeneity_score(labels_true, db.labels_)
        # comp = metrics.completeness_score(labels_true, db.labels_)
        sil = metrics.silhouette_score(test_data, db.labels_)
        if sil > best_sil:
            best_ep = ep
            best_sil = sil
            best_db = db

    core_samples_mask = np.zeros_like(best_db.labels_, dtype=bool)
    core_samples_mask[best_db.core_sample_indices_] = True
    labels = best_db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    results = {}
    results['best_epsilon'] = best_ep
    results['num_clusters'] = n_clusters_
    results['num_noise_points'] = n_noise_
    results['homogeneity_score'] = metrics.homogeneity_score(labels_true, labels)
    results['completeness'] = metrics.completeness_score(labels_true, labels)
    results['V-v_measure_score'] = metrics.v_measure_score(labels_true, labels)
    results['adjusted_rand_score'] = metrics.adjusted_rand_score(labels_true, labels)
    results['adjusted_mutual_information'] = metrics.adjusted_mutual_info_score(
        labels_true, labels, average_method='arithmetic')
    results['silhouette_score'] = metrics.silhouette_score(test_data, labels)
    return results

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

# master list of all occs involved:
test_occs = []
labels_true = []
for i, pn in enumerate(annPClassNames):
    occNames = pClasses[pn].occNames
    for on in occNames:
        test_occs.append(on)
        labels_true.append(i)

# add noisy occs:
for i in range(4):
    test_occs.append(str(np.random.choice(genPOccNames)))
    labels_true.append(-1)

# load saved model
model = nc.FFNetDistance(num_feats=len(sorted_fkeys))
model.load_state_dict(torch.load('saved_model.pt'))
model.eval()

res = evaluate_clustering(test_occs, labels_true, model, pOccs)
