import os
import pickle
import numpy as np
import itertools
import torch
import netClasses as nc
import prepareDataForTraining as pdft
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from importlib import reload
from collections import Counter
reload(pdft)
reload(nc)

pickle_name = 'parsed_patterns.pik'


def evaluate_clustering(test_occs, labels_true, model, pOccs, subset='all', eps_pctiles=None, reduce_with_pca=-1):
    fkeys = list(pOccs.values())[0].occFeatures.keys()
    sorted_fkeys = sorted(pdft.keys_subset(fkeys, subset))

    full_data = []
    for occ_name in test_occs:
        occ = pOccs[occ_name]
        arr = [occ.occFeatures[fkey] for fkey in sorted_fkeys]
        full_data.append(arr)

    if reduce_with_pca > 0:
        pca = PCA(reduce_with_pca)
        reduced_data = pca.fit_transform(np.array(full_data))
        full_data = reduced_data

    with torch.no_grad():
        full_data = torch.tensor(full_data).float()
        test_data = model.subspace(full_data).detach().numpy()

    return perform_dbscan(test_data, labels_true, eps_pctiles)


def evaluate_clustering_pca(test_occs, labels_true, pOccs, n_components=10, subset='all', eps_pctiles=None):
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
    return perform_dbscan(reduced_data, labels_true, eps_pctiles)


def perform_dbscan(test_data, labels_true, eps_pctiles=None, min_samples=3):

    labels_true = np.array(labels_true)
    n_clusters_true = len(set(labels_true)) - (1 if -1 in labels_true else 0)

    all_idxs = np.ones_like(labels_true, dtype='bool')
    in_noise_idxs = (labels_true != -1)

    all_label_sets = [all_idxs, in_noise_idxs]
    eps_to_try = estimate_best_epsilons(test_data, eps_pctiles, k=min_samples)

    labellings = []
    all_results = {}
    for ep_num, ep in enumerate(eps_to_try):
        db = DBSCAN(eps=ep, metric='l1', min_samples=min_samples).fit(test_data)
        n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)

        # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        # core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        labellings.append(labels)

        c = Counter(l for l in labels if l > 0)
        med_clust_size = np.median(list(c.values()))
        mean_clust_size = np.mean(list(c.values()))

        for i, idxs in enumerate(all_label_sets):
            res_str = 'eps {}, sig_only {}'.format(ep_num, i)
            l = labels[idxs]
            lt = labels_true[idxs]
            td = test_data[idxs]

            if n_clusters_ < 2:
                print('degenerate clustering')
                all_results[res_str] = False
                continue

            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(l)) - (1 if -1 in l else 0)
            n_noise_ = list(l).count(-1)
            results = {}
            results['epsilon'] = ep
            results['num_clusters'] = n_clusters_
            results['median_cluster_size'] = med_clust_size
            results['mean_cluster_size'] = mean_clust_size
            results['num_clusters_ratio'] = np.round(n_clusters_ / n_clusters_true, 3)
            results['num_noise_points'] = n_noise_
            results['homogeneity_score'] = np.round(metrics.homogeneity_score(lt, l), 3)
            results['completeness'] = np.round(metrics.completeness_score(lt, l), 3)
            results['V-v_measure_score'] = np.round(metrics.v_measure_score(lt, l), 3)
            results['adjusted_rand_score'] = np.round(metrics.adjusted_rand_score(lt, l), 3)
            try:
                results['silhouette_score'] = np.round(metrics.silhouette_score(td, l), 3)
            except ValueError:
                results['silhouette_score'] = -1
            all_results[res_str] = results
    return all_results, labellings


def estimate_best_epsilons(reduced_data, percentiles=None, k=3):
    if percentiles is None:
        percentiles = [90]

    nbrs = NearestNeighbors(n_neighbors=k, metric='l1').fit(reduced_data)
    distances, indices = nbrs.kneighbors(reduced_data)
    k_dist = sorted([x[-1] for x in distances])
    k_dist_pctiles = [np.percentile(k_dist, q) for q in percentiles]

    for i, val in enumerate(k_dist_pctiles):
        if val == 0:
            k_dist_pctiles[i] = min([x for x in k_dist_pctiles if x > 0]) / 2

    return k_dist_pctiles


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
    for i in range(len(genPOccNames) * 1):
        test_occs.append(str(genPOccNames[i]))
        labels_true.append(-1)

    # load saved model
    # model = nc.FFNetDistance(num_feats=len(sorted_fkeys), dim_size=10)
    # model.load_state_dict(torch.load('saved_model.pt'))
    # model.eval()
    # print('performing clustering...')
    # res = evaluate_clustering(test_occs, labels_true, pOccs, n_components=3, subset='all')
    res = evaluate_clustering_pca(test_occs, labels_true, pOccs, n_components=10, subset='all')
    print(res)
