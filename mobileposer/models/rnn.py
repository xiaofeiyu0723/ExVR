import torch.nn as nn
from torch.nn.functional import relu
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from mobileposer.config import *
import mobileposer.articulate as art


class RNN(torch.nn.Module):
    """
    A RNN Module including a linear input layer, a RNN and a linear output layer.
    """
    def __init__(self, n_input, n_output, n_hidden, n_rnn_layer=2, bidirectional=True, dropout=0.4):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(input_size=n_hidden, hidden_size=n_hidden, num_layers=n_rnn_layer, bidirectional=bidirectional)
        self.linear1 = nn.Linear(in_features=n_input, out_features=n_hidden)
        self.linear2 = nn.Linear(in_features=n_hidden * (2 if bidirectional else 1), out_features=n_output)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, seq_lengths=None, h=None):
        # pass input data through a linear layer
        data = self.dropout(relu(self.linear1(x)))
        # pack the padded sequences
        if seq_lengths is not None:
            data = pack_padded_sequence(data, seq_lengths, batch_first=True, enforce_sorted=False)
        # pass input to RNN
        data, h = self.rnn(data, h)
        # pack the padded sequences
        output_lengths = None
        if seq_lengths is not None:
            data, output_lengths = pad_packed_sequence(data, batch_first=True)
        data = self.linear2(data)
        return data, output_lengths, h
