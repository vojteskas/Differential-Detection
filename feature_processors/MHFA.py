import torch
import torch.nn as nn

from feature_processors.BaseProcessor import BaseProcessor

class MHFA(nn.Module, BaseProcessor):
    """
    Multi-head factorized attentive pooling layer.
    Implemented by Junyi Peng, 2022, based on the paper: https://arxiv.org/abs/2210.01273
    Code adapted from: https://github.com/JunyiPeng00/SLT22_MultiHead-Factorized-Attentive-Pooling/blob/master/models/Baseline/Spk_Encoder.py
    """

    def __init__(
        self, head_nb=32, input_transformer_nb=24, inputs_dim=1024, compression_dim=128, outputs_dim=1024
    ):
        """
        Initialize the MHFA feature processor.

        param head_nb: Number of attention heads
        param input_transformer_nb: Number (dimension) of transformer layers of the input features
        param inputs_dim: Dimension of the input features
        param compression_dim: Compressed features dim - each head will compress from inputs_dim to this dimension
        param outputs_dim: Dimension of the output features
        """
        super(MHFA, self).__init__()

        # Define learnable weights for key and value computations across layers
        self.weights_k = nn.Parameter(data=torch.ones(input_transformer_nb), requires_grad=True)
        self.weights_v = nn.Parameter(data=torch.ones(input_transformer_nb), requires_grad=True)

        # Initialize given parameters
        self.head_nb = head_nb
        self.ins_dim = inputs_dim
        self.cmp_dim = compression_dim
        self.out_dim = outputs_dim

        # Define compression linear layers for keys and values
        self.cmp_linear_k = nn.Linear(self.ins_dim, self.cmp_dim)
        self.cmp_linear_v = nn.Linear(self.ins_dim, self.cmp_dim)

        # Define linear layer to compute multi-head attention weights
        self.att_head = nn.Linear(self.cmp_dim, self.head_nb)

        # Define a fully connected layer for final output
        self.pooling_fc = nn.Linear(self.head_nb * self.cmp_dim, self.out_dim)

    def forward(self, x):
        # Input x has shape: [Nb_Layer, Batch, Frame_len, Dim]
        # Need to transpose to [Batch, Dim, Frame_len, Nb_Layer] for code reuse
        x = x.transpose(0, 1).transpose(1, 3)

        # Compute the key by taking a weighted sum of input across layers
        k = torch.sum(x.mul(nn.functional.softmax(self.weights_k, dim=-1)), dim=-1).transpose(1, 2)

        # Compute the value in a similar fashion
        v = torch.sum(x.mul(nn.functional.softmax(self.weights_v, dim=-1)), dim=-1).transpose(1, 2)

        # Pass the keys and values through compression linear layers
        k = self.cmp_linear_k(k)
        v = self.cmp_linear_v(v)

        # Compute attention weights using compressed keys
        att_k = self.att_head(k)

        # Adjust dimensions for computing attention output
        v = v.unsqueeze(-2)

        # Compute attention output by taking weighted sum of values using softmaxed attention weights
        pooling_outs = torch.sum(v.mul(nn.functional.softmax(att_k, dim=1).unsqueeze(-1)), dim=1)

        # Reshape the tensor before passing through the fully connected layer
        b, h, f = pooling_outs.shape
        pooling_outs = pooling_outs.reshape(b, -1)

        # Pass through fully connected layer to get the final output
        outs = self.pooling_fc(pooling_outs)

        return outs
