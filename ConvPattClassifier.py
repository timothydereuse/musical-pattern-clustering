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

use_cuda = True
batch_size = 400
num_categories = 2
learning_rate = 1e-4

class ConvNet(nn.Module):
    def __init__(self, img_size, out_size):
        super(ConvNet, self).__init__()
        layer1_chan = 50

        layer2_chan = 50
        hidden_layer = 500

        def get_new_size(self, old_size, padding, dilation, kernel_size, stride):
            h_out = (old_size[0] + (2 * padding[0]) - (dilation[0] * (kernel_size[0] - 1)) - 1) / (stride[0]) + 1
            v_out = (old_size[1] + (2 * padding[1]) - (dilation[1] * (kernel_size[1] - 1)) - 1) / (stride[1]) + 1
            return int(np.floor(h_out)), int(np.floor(v_out))

        # to start, input = img_size**2 * (1 channel)

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, layer1_chan, kernel_size=5, stride=1, padding=4, dilation=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # now input = (img_size / 2) * (layer1_chan)

        self.layer2 = nn.Sequential(
            nn.Conv2d(layer1_chan, layer2_chan, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        #self.layer1_size = get_new_size(layer1_size, kernel_size=5, stride=1, padding=4, dilation=2)

        self.drop_out = nn.Dropout()

        # now input = (img_size / 4)**2 * (layer2_chan)
        # not entirely sure yet how stride and padding affect the size of the layers;
        # changing them breaks the following line

        self.fc1 = nn.Linear((img_size[0] // 4) * (img_size[1] // 4) * layer2_chan, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, out_size)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out




# load some data
print('loading data...')
train_images, train_labels = apr.assemble_rolls()

proportion_positive = sum(train_labels) / len(train_labels)

img_size = train_images.shape[-2:]
num_train = len(train_images)

print('transforming data into tensors...')
# add dimension for channels
x = torch.tensor(train_images)[:, None, :, :].float()

# get labels into a tensor
y = torch.tensor(train_labels).long()


device = torch.device("cuda" if use_cuda else "cpu")


print('defining model...')
# define model
model = ConvNet(img_size=img_size, out_size=num_categories)
weights = torch.tensor([1, 1/proportion_positive])
loss_func = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
losses = []
accuracies = []

print('running model...')
for epoch in range(1000):

    # choose samples to use for this batch
    indices = np.random.choice(num_train, batch_size, replace=False)
    x_batch = x[indices]
    y_batch = y[indices]

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
    predicted = predicted.numpy()
    y_batch = y_batch.numpy()

    true_positive = sum((y_batch == 1) & (predicted == 1))
    true_negative = sum((y_batch == 0) & (predicted == 0))
    false_positive = sum((y_batch == 0) & (predicted == 1))
    false_negative = sum((y_batch == 1) & (predicted == 0))

    recall = true_positive / (true_positive + false_negative + 1)
    precision = true_positive / (true_positive + false_positive + 1)
    if true_positive:
        F1 = 2 * (precision * recall) / (recall + precision)
    else:
        F1 = 0

    accuracies.append(np.round((recall, precision, F1), 4))
    accuracy = accuracies[-1]

    rd_loss = np.round(loss.item(), 4)

    print(f"Epoch : {epoch}    Loss : {rd_loss}    F1:{accuracy}")
