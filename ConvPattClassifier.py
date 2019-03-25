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
reload(apr)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
num_validation_sets = 6
num_categories = 2
learning_rate = 2e-4

class ConvNet(nn.Module):
    def __init__(self, img_size, out_size):
        super(ConvNet, self).__init__()
        layer1_chan = 100
        layer2_chan = 100
        hidden_layer = 500

        conv1_kwargs = {'kernel_size':5, 'stride':1, 'padding':(0,4), 'dilation':1}
        max1_kwargs = {'kernel_size':2, 'stride':2}

        conv2_kwargs = {'kernel_size':5, 'stride':1, 'padding':(0,4), 'dilation':1}
        max2_kwargs = max1_kwargs

        def get_new_size(old_size, kernel_size=3, padding=0, dilation=1, stride=None):

            if type(kernel_size) is not tuple:
                kernel_size = (kernel_size, kernel_size)
            if not stride:
                stride = kernel_size
            if type(stride) is not tuple:
                stride = (stride, stride)
            if type(padding) is not tuple:
                padding = (padding, padding)
            if type(dilation) is not tuple:
                dilation = (dilation, dilation)

            # print(kernel_size, stride, padding, dilation)

            h_out = (old_size[0] + (2 * padding[0]) - (dilation[0] * (kernel_size[0] - 1)) - 1) / (stride[0]) + 1
            v_out = (old_size[1] + (2 * padding[1]) - (dilation[1] * (kernel_size[1] - 1)) - 1) / (stride[1]) + 1
            return int(np.floor(h_out)), int(np.floor(v_out))

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, layer1_chan, **conv1_kwargs),
            nn.ReLU(),
            nn.MaxPool2d(**max1_kwargs),
            nn.BatchNorm2d(layer1_chan)
        )
        self.drop_out_2d = nn.Dropout2d()

        self.layer2 = nn.Sequential(
            nn.Conv2d(layer1_chan, layer2_chan, **conv2_kwargs),
            nn.ReLU(),
            nn.MaxPool2d(**max2_kwargs),
            nn.BatchNorm2d(layer1_chan)
        )
        self.drop_out = nn.Dropout()

        new_size = get_new_size(img_size, **conv1_kwargs)
        new_size = get_new_size(new_size, **max1_kwargs)
        new_size = get_new_size(new_size, **conv2_kwargs)
        new_size = get_new_size(new_size, **max2_kwargs)

        self.fc1 = nn.Linear(new_size[0] * new_size[1] * layer2_chan, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, out_size)

    def forward(self, x):
        out = self.layer1(x)
        out = self.drop_out_2d(out)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

def train_model(dataset, labels, device, batch_size=None, num_epochs=1000, stagnation_time=100):

    x = dataset
    y = labels

    # define model
    model = ConvNet(img_size=img_size, out_size=num_categories).to(device)
    weights = torch.tensor([1, 1/proportion_positive], device=device)
    loss_func = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    accuracies = []
    best_f1 = 0
    epocs_since_best_f1 = 0
    print('running model...')
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
train_images, train_labels = apr.assemble_rolls(normalize=True)

proportion_positive = sum(train_labels) / len(train_labels)

img_size = train_images.shape[-2:]
num_train = len(train_images)

# manage validation sets
idx_shuffle = np.array(range(num_train))
np.random.shuffle(idx_shuffle)
set_idxs = np.array_split(idx_shuffle, num_validation_sets)

train_idxs = np.concatenate(set_idxs[:-1])
test_idxs = set_idxs[-1]

print('transforming data into tensors...')
# add dimension for channels
x_all = torch.tensor(train_images)[:, None, :, :].float()
x_train = x_all[train_idxs]
x_test = x_all[test_idxs]

# get labels into a tensor
y_all = torch.tensor(train_labels).long()
y_train = y_all[train_idxs]
y_test = y_all[test_idxs]

mod = train_model(x_train, y_train, device, batch_size=600, num_epochs=900)
mod = mod.cpu()

with torch.no_grad():
    output = mod(x_test)
    preds = output.argmax(dim=1)

results = calculate_stats(y_test.numpy(), preds.numpy())
print(results)
