#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dnn import Net
from src.phones import TextParser

text_parser = TextParser()
no_tokens = text_parser.get_no_tokens()
data_length = 1010


# Start timer
timer_start = time.time()

# Create neural network
net = Net().double().cuda()
net.load_state_dict(torch.load('model.pth'))
print('NET CREATED')

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    net.parameters(),
    lr=0.01,
    momentum=0.9
)

# Train
# try:
#     os.remove('train.log')
# except OSError:
#     pass
corpus_path = './LJSpeech-1.1'
for epoch in range(1):

    for i, filename in enumerate(
        os.listdir(corpus_path + '/output-train')
    ):

        # Load input data
        input_data = np.fromfile(
            corpus_path + '/input/' + filename,
            dtype=float
        )

        # Split input data to mfcc and tokens
        input_data = input_data.reshape((14, -1))

        # Load output data
        output_data = np.fromfile(
            corpus_path + '/output-train/' + filename,
            dtype=float
        )
        output_data = torch.from_numpy(
            output_data
        ).long().cuda()

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Predict output
        predicted_outputs = net(input_data)
        predicted_outputs = predicted_outputs[:, 0, :]

        # Compute loss
        loss = criterion(predicted_outputs, output_data)

        # Optimize
        loss.backward()
        optimizer.step()

        # Print statistics
        file_loss = loss.item()
        end_timer = time.time()
        if i % 10 == 0:
            print(
                'EPOCH',
                epoch,
                ', FILE',
                i,
                ', LOSS',
                file_loss,
                ', TIME',
                end_timer - timer_start
            )
        with open('train.log', 'a') as log_file:
            log_file.write(
                'EPOCH '\
                + str(epoch)\
                + ' , FILE '\
                + str(i)\
                + ' , LOSS '\
                + str(file_loss)\
                + ' , TIME '\
                + str(end_timer - timer_start)\
                + '\n'
            )

print('TRAINING FINISHED')
torch.save(net.state_dict(), 'model.pth')
print('MODEL SAVED')
