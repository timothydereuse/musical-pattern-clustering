# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 13:53:59 2018

@author: Tim
"""

from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt
import pickle
import numpy as np

import tensorflow as tf
import tensorflow.contrib.eager as tfe

#tf.enable_eager_execution()

print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

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

#set up dictionary of features
pClassFeatureKeys = pClasses[annPClassNames[0]]['classFeatures'].keys()
pClassFeatureKeys = sorted(pClassFeatureKeys)

featDict = {}
labels = []
for k in (pClassFeatureKeys[70:75]):
    temp = [ tf.to_float(pClasses[cn]['classFeatures'][k])
     for cn in (filtGenPClassNames + annPClassNames)]
    featDict[k] = np.array(temp)

labels = [pClasses[cn]['type'] == 'ann' 
          for cn in (filtGenPClassNames + annPClassNames)]
labels = np.array([tf.to_int32(x) for x in labels])

dataset = tf.data.Dataset.from_tensor_slices((featDict,labels))
dataset = dataset.shuffle(1000).repeat().batch(2)

#train_dataset = tf.data.TextLineDataset(train_dataset_fp)
#train_dataset = train_dataset.skip(1)             # skip the first header row
#train_dataset = train_dataset.map(parse_csv)      # parse each row
#train_dataset = train_dataset.shuffle(buffer_size=1000)  # randomize
#train_dataset = train_dataset.batch(32)

features, label = tfe.Iterator(dataset).next()

model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation="relu", 
                        input_shape=(5,)),  # input shape required
  tf.keras.layers.Dense(10, activation="relu"),
  tf.keras.layers.Dense(1)
])
    
def loss(model, x, y):
  y_ = model(x)
  return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

def grad(model, inputs, targets):
  with tfe.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return tape.gradient(loss_value, model.variables)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)


## Note: Rerunning this cell uses the same model variables
# keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 201

for epoch in range(num_epochs):
  epoch_loss_avg = tfe.metrics.Mean()
  epoch_accuracy = tfe.metrics.Accuracy()

  # Training loop - using batches of 32
  for x, y in tfe.Iterator(dataset):
    # Optimize the model
    grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.variables),
                              global_step=tf.train.get_or_create_global_step())

    # Track progress
    epoch_loss_avg(loss(model, x, y))  # add current batch loss
    # compare predicted label to actual label
    epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

  # end epoch
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())

  if epoch % 50 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))
    
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)

plt.show()