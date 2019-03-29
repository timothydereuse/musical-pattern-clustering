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


class FFNet(nn.Module):
    def __init__(self, num_feats, out_size):
        super(FFNet, self).__init__()
        layer1_chan = 50
        layer2_chan = 50
        layer3_chan = 50
        drop_out_prob = 0.50

        self.drop_out = nn.Dropout(p=drop_out_prob)
        self.fc1 = nn.Linear(num_feats, layer1_chan)
        self.norm_1 = nn.BatchNorm1d(layer1_chan)
        self.fc2 = nn.Linear(layer1_chan, layer2_chan)
        self.norm_2 = nn.BatchNorm1d(layer2_chan)
        self.fc3 = nn.Linear(layer2_chan, out_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.norm_1(out)
        out = self.drop_out(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.norm_2(out)
        out = self.drop_out(out)
        out = self.fc3(out)
        return out


class FFNetDistance(nn.Module):
    def __init__(self, num_feats):
        super(FFNetDistance, self).__init__()
        layer1_chan = 50
        layer2_chan = 50
        layer3_chan = 50
        drop_out_prob = 0.50

        self.layer1 = nn.Sequential(
            nn.Linear(num_feats, layer1_chan),
            nn.ReLU(),
            nn.BatchNorm1d(layer1_chan),
            nn.Dropout(p=drop_out_prob)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(layer1_chan, layer2_chan),
            nn.ReLU(),
            nn.BatchNorm1d(layer2_chan),
            nn.Dropout(p=drop_out_prob)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(layer2_chan, layer3_chan)
        )

        self.sigmoid = nn.Sigmoid()
        self.dist = nn.PairwiseDistance(p=1.0)

    def forward(self, x):
        f0 = x[:, 0, :]
        f1 = x[:, 1, :]

        out = self.layer1(f0)
        out = self.layer2(out)
        res1 = self.layer3(out)

        out = self.layer1(f1)
        out = self.layer2(out)
        res2 = self.layer3(out)

        distance = self.dist(res1, res2)
        return distance


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


if __name__ == '__main__':
    # put garbage data thru a net to test it
    learning_rate = 2e-4
    num_feats = 20
    batch_size = 300
    epochs = 500
    model = FFNetDistance(num_feats)
    loss_func = nn.HingeEmbeddingLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    x = torch.rand(batch_size, 2, num_feats)
    y = torch.randint(0, 2, (batch_size,)) * 2 - 1

    for epoch in range(epochs):
        y_pred = model(x)
        loss = loss_func(y_pred, y)

        # Reset gradients to zero, perform a backward pass, and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rd_loss = np.round(loss.item(), 4)
        print(f"Epoch : {epoch}    Loss : {rd_loss}")
