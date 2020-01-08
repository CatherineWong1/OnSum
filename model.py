# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn


class Summary(nn.Module):
    def __init__(self, batch_size, max_len):
        super(Summary, self).__init__()
        self.batch_size = batch_size
        self.max_len = max_len
        self.linear = nn.Linear(max_len, max_len, bias=False)
        self.bias = nn.Parameter(torch.randn(self.batch_size, self.max_len))

    def forward(self, batch_df):
        # linear output: (batch_size, max_len)
        linear_output = self.linear(batch_df) + self.bias
        # binary_output: (batch_size, max_len)
        self.binary_output = torch.sigmoid(linear_output)

        return self.binary_output
