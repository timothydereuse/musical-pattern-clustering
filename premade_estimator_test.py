#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""An Example of a DNNClassifier for the Iris dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pickle
import numpy as np

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
    # Create three fully connected layers each layer having a dropout
    # probability of 0.1.
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

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

pClassFeatureKeys = pClasses[annPClassNames[0]]['classFeatures'].keys()
pClassFeatureKeys = sorted(pClassFeatureKeys)

trainSteps = 500
batchSize = 100

np.random.shuffle(annPClassNames)
np.random.shuffle(filtGenPClassNames)
splitPos = int(round(len(annPClassNames) * 0.2))
valPClassNames = annPClassNames[splitPos:] + filtGenPClassNames[splitPos:]
testPClassNames = annPClassNames[:splitPos] + filtGenPClassNames[:splitPos]

rawTrainFeats = {}
rawTrainLabels = []
rawTestFeats = {}
rawTestLabels = []
for k in (pClassFeatureKeys):
    temp = [ pClasses[cn]['classFeatures'][k]
     for cn in (valPClassNames)]
    rawTrainFeats[k] = np.array(temp)
    
    temp = [ pClasses[cn]['classFeatures'][k]
     for cn in (testPClassNames)]
    rawTestFeats[k] = np.array(temp)

rawTrainLabels = [pClasses[cn]['type'] == 'ann' 
          for cn in (valPClassNames)]
rawTrainLabels = np.array([int(x) for x in rawTrainLabels])

rawTestLabels = [pClasses[cn]['type'] == 'ann' 
          for cn in (testPClassNames)]
rawTestLabels = np.array([int(x) for x in rawTestLabels])



my_feature_columns = []
for key in pClassFeatureKeys:
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

# Build 2 hidden layer DNN with 10, 10 units respectively.
#classifier = tf.estimator.DNNClassifier(
#    feature_columns=my_feature_columns,
#    # Two hidden layers of 10 nodes each.
#    hidden_units=[4, 4],
#    # The model must choose between 3 classes.
#    n_classes=2)

classifier = tf.estimator.Estimator(
    model_fn=my_model,
    params={
        'feature_columns': my_feature_columns,
        'hidden_units': [5, 5],
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

## Generate predictions from the model
#expected = ['Setosa', 'Versicolor', 'Virginica']
#predict_x = {
#    'SepalLength': [5.1, 5.9, 6.9],
#    'SepalWidth': [3.3, 3.0, 3.1],
#    'PetalLength': [1.7, 4.2, 5.4],
#    'PetalWidth': [0.5, 1.5, 2.1],
#}
#
#predictions = classifier.predict(
#    input_fn=lambda:iris_data.eval_input_fn(predict_x,
#                                            labels=None,
#                                            batch_size=batchSize))
#
#template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')
#
#for pred_dict, expec in zip(predictions, expected):
#    class_id = pred_dict['class_ids'][0]
#    probability = pred_dict['probabilities'][class_id]
#
#    print(template.format(iris_data.SPECIES[class_id],
#                          100 * probability, expec))


#if __name__ == '__main__':
#    tf.logging.set_verbosity(tf.logging.INFO)
#    tf.app.run(main)
