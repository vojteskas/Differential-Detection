from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_processors.BaseProcessor import BaseProcessor


class SLS(nn.Module, BaseProcessor):
    """
    Sensitive Layer Selection
    Implemented by Qishan Zhang et al., 2024, based on the paper: https://dl.acm.org/doi/pdf/10.1145/3664647.3681345
    Code adapted from: https://github.com/QiShanZhang/SLSforASVspoof-2021-DF
    """

    def __init__(
        self, inputs_dim=1024, outputs_dim=1024
    ):
        """
        Initialize the SLS feature processor.

        param inputs_dim: Dimension of the input features
        param outputs_dim: Dimension of the output features
        """
        super(SLS, self).__init__()

        # Initialize given parameters
        self.ins_dim = inputs_dim
        self.out_dim = outputs_dim

        # Calculate the hidden layer size after 3x3 pooling
        hidden_dim = ceil((inputs_dim - 2) / 3) * ceil((201 - 2) / 3) # 22847 for 1024 features (201 frames)
        # print(f"Hidden dim: {hidden_dim}")

        # Initialize layers
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.fc0 = nn.Linear(inputs_dim, 1)
        self.sig = nn.Sigmoid()
        self.fc1 = nn.Linear(hidden_dim, outputs_dim)
        
        # Not used, we use our own classifier
        # self.fc3 = nn.Linear(1024,2)
        # self.logsoftmax = nn.LogSoftmax(dim=1)

    def getAttenF(self, layerResult):
        poollayerResult = []
        fullf = []
        for layer in layerResult:
            layery = layer.transpose(0, 1).transpose(1, 2) #(x,z)  x(201,b,1024) (b,201,1024) (b,1024,201)
            layery = F.adaptive_avg_pool1d(layery, 1) #(b,1024,1)
            layery = layery.transpose(1, 2) # (b,1,1024)
            poollayerResult.append(layery)

            x = layer.transpose(0, 1)
            x = x.view(x.size(0), -1,x.size(1), x.size(2))
            fullf.append(x)

        layery = torch.cat(poollayerResult, dim=1)
        fullfeature = torch.cat(fullf, dim=1)
        return layery, fullfeature

    def forward(self, x):
        # Cull or pad the input to 4 seconds (201 frames)
        if x.shape[2] < 201:
            x = F.pad(x, (0, 0, 201 - x.shape[2], 0))
        elif x.shape[2] > 201:
            x = x[:, :, :201, :]

        # Input x has shape: [Nb_Layer, Batch, Frame_len, Dim]
        # Need to transpose to [Nb_Layer, Frame_len, Batch, Dim]
        x = x.transpose(1, 2)

        # compute H from the paper
        y0, fullfeature = self.getAttenF(x)

        # the upper branch in figure 3 in the paper
        y0 = self.fc0(y0)
        y0 = self.sig(y0)
        y0 = y0.view(y0.shape[0], y0.shape[1], y0.shape[2], -1)

        # the lower branch in figure 3 in the paper
        fullfeature = fullfeature * y0
        fullfeature = torch.sum(fullfeature, 1)
        fullfeature = fullfeature.unsqueeze(dim=1)

        # classifier part (red and blue from figure 2 of the paper)
        x = self.first_bn(fullfeature)
        x = self.selu(x)
        x = F.max_pool2d(x, (3, 3))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.selu(x)

        # last classification layer, we use our classifiers
        # x = self.fc3(x)
        # x = self.selu(x)
        # output = self.logsoftmax(x)

        return x
