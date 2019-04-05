import os
import pickle
import numpy as np
import itertools
import torch
import netClasses as nc
import prepareDataForTraining as pdft
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.decomposition import PCA
from importlib import reload
reload(pdft)
reload(nc)

pickle_name = 'parsed_patterns.pik'


def evaluate_clustering(test_occs, labels_true, model, pOccs, subset='all', epsilons=None):
    fkeys = list(pOccs.values())[0].occFeatures.keys()
    sorted_fkeys = sorted(pdft.keys_subset(fkeys, subset))

    full_data = []
    for occ_name in test_occs:
        occ = pOccs[occ_name]
        arr = [occ.occFeatures[fkey] for fkey in sorted_fkeys]
        full_data.append(arr)

    with torch.no_grad():
        full_data = torch.tensor(full_data).float()
        test_data = model.subspace(full_data).detach().numpy()

    return perform_dbscan(test_data, labels_true, epsilons)


def evaluate_clustering_pca(test_occs, labels_true, pOccs, n_components=10, subset='all', epsilons=None):
    '''
    use dimensionality reduction with no distance learning, cluster data, and see how well
    the found clusters match up with ground truth.
    '''
    fkeys = list(pOccs.values())[0].occFeatures.keys()
    sorted_fkeys = sorted(pdft.keys_subset(fkeys, subset))

    full_data = []
    for occ_name in test_occs:
        feats = [pOccs[occ_name].occFeatures[fkey] for fkey in sorted_fkeys]
        full_data.append(feats)

    pca = PCA(n_components)
    reduced_data = pca.fit_transform(full_data)
    return perform_dbscan(reduced_data, labels_true, epsilons)


def perform_dbscan(test_data, labels_true, epsilons=None):

    if epsilons is None:
        eps_to_try = np.geomspace(1e-6, 5e-1, 100)
    elif not (hasattr(epsilons, '__iter__')):
        eps_to_try = [epsilons]
    else:
        eps_to_try = epsilons

    best_db = None
    best_ep = 0
    best_sil = -1
    for ep in eps_to_try:
        db = DBSCAN(eps=ep, metric='l1', min_samples=3).fit(test_data)
        # hom = metrics.homogeneity_score(labels_true, db.labels_)
        # comp = metrics.completeness_score(labels_true, db.labels_)
        n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        if n_clusters_ < 2:
            continue
        sil = metrics.silhouette_score(test_data, db.labels_)
        if sil > best_sil:
            best_ep = ep
            best_sil = sil
            best_db = db

    if best_sil == -1:
        return False

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
    results['silhouette_score'] = metrics.silhouette_score(test_data, labels)
    return results


def estimate_best_epsilon(data, model):
    x_val = data[0]
    y_val = data[1]
    similar_idxs = [n for n in range(len(x_val)) if y_val[n] == 1]

    similar_data = x_val[similar_idxs]
    with torch.no_grad():
        distances = model(similar_data).detach().numpy()

    avg_distance = np.median(distances)
    return avg_distance


if __name__ == '__main__':
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
    for i in range(len(test_occs)):
        test_occs.append(str(np.random.choice(genPOccNames)))
        labels_true.append(-1)

    # load saved model
    # model = nc.FFNetDistance(num_feats=len(sorted_fkeys))
    # model.load_state_dict(torch.load('saved_model.pt'))
    # model.eval()
    print('performing clustering...')
    res = evaluate_clustering_pca(test_occs, labels_true, pOccs, n_components=5, subset='only_seq')
    print(res)
