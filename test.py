#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dnn import Net
from src.phones import TextParser

text_parser = TextParser()
no_tokens = text_parser.get_no_tokens()
data_length = 1010


# Create neural network
net = Net().double().cuda()
net.load_state_dict(torch.load('model.pth'))
print('NET LOADED')

# Test
try:
    os.remove('test.log')
except OSError:
    pass
corpus_path = './LJSpeech-1.1'
error_matrix = np.zeros((no_tokens, no_tokens))
for i, filename in enumerate(
    os.listdir(corpus_path + '/output-test')
):

    # Load test data
    input_data = np.fromfile(
        corpus_path + '/input/' + filename,
        dtype=float
    )

    # Split input data to mfcc and tokens
    input_data = input_data.reshape((14, -1))

    # Load output data
    output_data = np.fromfile(
        corpus_path + '/output-test/' + filename,
        dtype=float
    )
    output_data = torch.from_numpy(
        output_data
    ).long().cuda()

    # Predict output
    predicted_outputs = net(input_data)
    predicted_outputs = predicted_outputs[:, 0, :]
    _, predicted_outputs = torch.max(predicted_outputs, 1)

    # Update error matrix
    for j in range(output_data.shape[0]):
        error_matrix[output_data[j]][predicted_outputs[j]] += 1
    if i % 10 == 0:
        print(i, 'TESTED')

# Calculate ver
all_answers = np.sum(error_matrix)
correct_answers = np.sum(
    error_matrix * np.identity(no_tokens)
)
ver = (all_answers - correct_answers)/all_answers

# Print statistics
print('TESTING FINISHED')
print('VITERBI ERROR RATE =', ver)
with open('test.log', 'a') as log_file:
            log_file.write(
                'VITERBI ERROR RATE = '\
                + str(ver*100)\
                + '%\n'
            )
plt.matshow(error_matrix)
plt.colorbar()
plt.show()
