# python patternClassifierNN.py all; python patternClassifierNN.py exclude_stds; python patternClassifierNN.py exclude_means; python patternClassifierNN.py exclude_song_comp; python patternClassifierNN.py only_pitch; python patternClassifierNN.py only_rhythm;

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import tensorflow as tf
import featureExtractors as ft
import pickle
import numpy as np
import sys

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset

def my_model(features, labels, mode, params):
    """DNN with three hidden layers, and dropout of 0.1 probability."""

    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.sigmoid)

    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def train_and_test_knn(num_run, num_chunks, feature_subset, data_sets, pClasses):
    trainSteps = 1000
    batchSize = 100

    pClassFeatureKeys = list(pClasses[list(pClasses.keys())[0]].classFeatures.keys())
    subset = ft.keys_subset(pClassFeatureKeys,feature_subset)

    #get current test and train sets from the run num
    test_pat_names = data_sets[num_run]
    val_pat_names = [data_sets[i] for i in range(num_chunks) if i is not num_run]
    val_pat_names = [item for sublist in val_pat_names for item in sublist]

    # need to get a dictionary where every feature is a key, and every value is
    # an array containing an ordered list of feature values for the current set.
    for k in (subset):
        temp = [pClasses[cn].classFeatures[k] for cn in (val_pat_names)]
        rawTrainFeats[k] = np.array(temp)

        temp = [pClasses[cn].classFeatures[k] for cn in (test_pat_names)]
        rawTestFeats[k] = np.array(temp)

    #the labels are the same thing, but just an array.
    rawTrainLabels = [pClasses[cn].type == 'ann' for cn in (val_pat_names)]
    rawTrainLabels = np.array([int(x) for x in rawTrainLabels])

    rawTestLabels = [pClasses[cn].type == 'ann' for cn in (test_pat_names)]
    rawTestLabels = np.array([int(x) for x in rawTestLabels])

    #my_feature_columns is just a list of keys
    my_feature_columns = []
    for key in subset:
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params={
            'feature_columns': my_feature_columns,
            'hidden_units': [6,6],
            'n_classes': 2,
        })

    # Train the Model.
    classifier.train(
        input_fn=lambda:train_input_fn(rawTrainFeats, rawTrainLabels,
                                       batchSize),
        steps=trainSteps)

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:eval_input_fn(rawTestFeats, rawTestLabels,
                                                batchSize))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
    return eval_result['accuracy']

if __name__ == "__main__":

    num_chunks = 5
    feature_subset = 'all'
    #feature_subsets = ['all','only_pitch','only_rhythm','exclude_stds',
    #               'exclude_means','exclude_song_comp']

    if (len(sys.argv) > 1):
        feature_subset = sys.argv[1]

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

    pClassFeatureKeys = pClasses[annPClassNames[0]].classFeatures.keys()
    pClassFeatureKeys = sorted(pClassFeatureKeys)

    np.random.shuffle(annPClassNames)
    np.random.shuffle(filtGenPClassNames)

    #combine annotated and generated names into dataset containing 5 folds.
    ann_chunks = ft.split_into_chunks(annPClassNames,num_chunks)
    gen_chunks = ft.split_into_chunks(filtGenPClassNames,num_chunks)
    data_sets = [ann_chunks[i] + gen_chunks[i] for i in range(num_chunks)]

    rawTrainFeats = {}
    rawTrainLabels = []
    rawTestFeats = {}
    rawTestLabels = []

    currentTime = str(datetime.datetime.now())
    filename = "NN RUN %s,%s 500 runs.txt" % (feature_subset,currentTime)
    filename = filename.replace(":","-")

    accs = []
    for num_run in range(num_chunks):
        res = train_and_test_knn(num_run, num_chunks, feature_subset,
                                    data_sets, pClasses)
        accs.append(res)

    # file = open(filename,"a")
    # file.write("all: %s \n" % str(accs))
    # file.write("mean: %s \n" % np.mean(accs))
    # file.write("std: %s \n" % np.std(accs))
    # tmp = np.std(accs) / np.sqrt(num_chunks)
    # file.write("stderr: %s \n" % tmp)
    # file.close()
