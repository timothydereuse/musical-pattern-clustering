import pickle as pkl
import torch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from importlib import reload

import assemblePianoRolls as apr
import netClasses as nc
reload(apr)
reload(nc)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
num_validation_sets = 4
num_categories = 2
learning_rate = 2e-4


def train_model(dataset, labels, model, device, batch_size=None, num_epochs=1000, stagnation_time=500):

    proportion_positive = (sum(labels) + 1) / (len(labels) + 1)

    x = dataset
    y = labels

    # define model

    weights = torch.tensor([1.0, 10.0], device=device)
    loss_func = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    accuracies = []
    best_f1 = 0
    epocs_since_best_f1 = 0
    for epoch in range(num_epochs):

        # choose samples to use for this batch
        if batch_size:
            indices = np.random.choice(x.shape[0], batch_size, replace=False)
        else:
            indices = np.array(range(len(x)))
        x_batch = x[indices].to(device)
        y_batch = y[indices].to(device)

        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x_batch)

        # Compute loss
        loss = loss_func(y_pred, y_batch)
        losses.append(loss.item())

        # Reset gradients to zero, perform a backward pass, and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        total = y_batch.size(0)
        _, predicted = torch.max(y_pred.data, 1)
        predicted = predicted.cpu().numpy()

        stats = calculate_stats(y[indices].numpy(), predicted)
        accuracies.append(stats)
        rd_loss = np.round(loss.item(), 4)
        if not epoch % 100:
            print(f"Epoch : {epoch}    Loss : {rd_loss}    rec/prec/f1:{stats}")

        if stats[2] > best_f1:
            best_f1 = stats[2]
            epocs_since_best_f1 = 0
        else:
            epocs_since_best_f1 += 1

        if epocs_since_best_f1 >= stagnation_time:
            break

    return model


def calculate_stats(correct, predicted, round_to=3):
    true_positive = sum((correct == 1) & (predicted == 1))
    true_negative = sum((correct == 0) & (predicted == 0))
    false_positive = sum((correct == 0) & (predicted == 1))
    false_negative = sum((correct == 1) & (predicted == 0))

    recall = true_positive / (true_positive + false_negative + 1)
    precision = true_positive / (true_positive + false_positive + 1)
    if true_positive:
        F1 = 2 * (precision * recall) / (recall + precision)
    else:
        F1 = 0

    ret = np.round((recall, precision, F1), round_to)
    return ret


# load some data
print('loading data...')
# train_images, train_labels = apr.assemble_rolls(normalize=True)
train_images, train_labels = apr.assemble_feats()


img_size = train_images.shape[-2:]
num_train = len(train_images)

num_dimensions = len(train_images.shape)
if num_dimensions == 3:
    x_all = torch.tensor(train_images)[:, None, :, :].float()
elif num_dimensions == 2:
    x_all = torch.tensor(train_images)[:, :].float()

# get labels into a tensor
y_all = torch.tensor(train_labels).long()

# manage validation sets
idx_shuffle = np.array(range(num_train))
np.random.shuffle(idx_shuffle)
set_idxs = np.array_split(idx_shuffle, num_validation_sets)
quit()
cross_val_results = []
for run_num in range(num_validation_sets):

    train_idxs = np.concatenate(set_idxs[:run_num] + set_idxs[run_num+1:])
    test_idxs = set_idxs[run_num]

    # add dimension for channels
    x_train = x_all[train_idxs]
    x_test = x_all[test_idxs]

    y_train = y_all[train_idxs]
    y_test = y_all[test_idxs]

    # model = nc.ConvNet(img_size=img_size, out_size=num_categories)
    model = nc.FFNet(num_feats=x_all.shape[-1], out_size=2)

    print('running model...')
    mod = train_model(x_train, y_train, model, device,
        batch_size=300, num_epochs=5000, stagnation_time=1000)
    mod = mod.cpu()

    with torch.no_grad():
        output = mod(x_test)
        preds = output.argmax(dim=1)

    results = calculate_stats(y_test.numpy(), preds.numpy())
    cross_val_results.append(results)
    print(results)

for i in range(len(cross_val_results[0])):
    print(np.mean([x[i] for x in cross_val_results]))
