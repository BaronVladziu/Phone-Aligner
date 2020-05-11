#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.phones import TextParser

text_parser = TextParser()
no_tokens = text_parser.get_no_tokens()
data_length = 1010
embedding_dim = 16

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.emb = nn.Embedding(
            num_embeddings=no_tokens,
            embedding_dim=embedding_dim
        )
        self.lstm = nn.LSTM(
            input_size=13 + embedding_dim,
            hidden_size=no_tokens,
            num_layers=5,
            bidirectional=True
        )
        self.lin = nn.Linear(
            in_features=2*no_tokens,
            out_features=no_tokens
        )

    def forward(self, x):
        # Use embedding on tokens
        input_tokens = x[13]
        input_tokens = torch.from_numpy(
            input_tokens
        ).long().cuda().reshape((-1, 1))
        input_tokens = self.emb(input_tokens)

        # Concatenate tokens with feats
        input_fts = x[:13]
        input_fts = torch.from_numpy(
            input_fts
        ).double().cuda().reshape((-1, 1, 13))
        x = torch.cat([input_fts, input_tokens], 2)

        # Process
        x, _ = self.lstm(x)
        x = self.lin(x)
        return x
