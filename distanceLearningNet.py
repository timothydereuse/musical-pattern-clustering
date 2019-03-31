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

import prepareDataForTraining as apr
import netClasses as nc
reload(apr)
reload(nc)

device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_validation_sets = 15
learning_rate = 3e-4


def train_model(data, model, device, batch_size=None, num_epochs=1000, stagnation_time=500, val_data=None, poll_every=50):

    x, y = data

    if val_data:
        val_x, val_y = val_data

    loss_func = nn.HingeEmbeddingLoss(reduction='mean')
    eval_loss_func = nn.functional.hinge_embedding_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    accuracies = []
    best_loss = 0
    epocs_since_best_loss = 0
    for epoch in range(int(num_epochs)):

        # choose samples to use for this batch
        if batch_size:
            indices = np.random.choice(x.shape[0], batch_size, replace=False)
        else:
            indices = np.array(range(x.shape[0]))
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

        rd_loss = np.round(loss.item(), 4)

        if epoch % poll_every:
            continue

        if val_data:
            with torch.no_grad():
                y_val_pred = model(val_data[0])
            val_loss = eval_loss_func(y_val_pred, val_data[1], reduction='none')
        else:
            val_loss = eval_loss_func(y_pred, y_batch, reduction='none')
        # accuracies.append(val_loss.item())
        num_correct = sum(val_loss < 1).item()
        accuracy = np.round(num_correct / len(val_loss), 4)
        accuracies.append(accuracy.item())

        print("Epoch: {}    Loss: {:.2E},    Accuracy:{:.2E}".format(
            epoch, rd_loss, accuracy))

        if sum(val_loss) > best_loss:
            best_loss = sum(val_loss)
            epocs_since_best_loss = 0
        else:
            epocs_since_best_loss += poll_every

        if epocs_since_best_loss >= stagnation_time:
            break

    return model, accuracies


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
train_images, train_labels = apr.assemble_clustering_feats(
    unsimilar_factor=5, gen_factor=5, max_similar=0)

num_train = len(train_images)

x_all = torch.tensor(train_images).float()
y_all = torch.tensor(train_labels).long()

# manage validation sets
idx_shuffle = np.array(range(num_train))
np.random.shuffle(idx_shuffle)
set_idxs = np.array_split(idx_shuffle, num_validation_sets)

cross_val_results = []
for run_num in range(1):  # range(num_validation_sets):

    train_idxs = np.concatenate(set_idxs[:run_num] + set_idxs[run_num+1:])
    test_idxs = set_idxs[run_num]

    # add dimension for channels
    x_train = x_all[train_idxs]
    x_val = x_all[test_idxs]

    y_train = y_all[train_idxs]
    y_val = y_all[test_idxs]

    # model = nc.ConvNet(img_size=img_size, out_size=num_categories)
    model = nc.FFNetDistance(num_feats=x_all.shape[-1])
    model.to(device)

    print('running model...')
    mod, accs = train_model((x_train, y_train), model, device, batch_size=256,
        num_epochs=1e6, stagnation_time=1e4, poll_every=2500, val_data=(x_val, y_val))
    # mod = mod.cpu()

    # with torch.no_grad():
    #     output = mod(x_test)
    #     preds = output.argmax(dim=1)
    #
    # results = calculate_stats(y_test.numpy(), preds.numpy())
    # cross_val_results.append(results)
    # print(results)

print(accs)
torch.save(mod.state_dict(),'saved_model.pt')

# for i in range(len(cross_val_results[0])):
#     print(np.mean([x[i] for x in cross_val_results]))
