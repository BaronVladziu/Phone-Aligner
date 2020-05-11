#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil

in_path = os.environ['HOME'] + '/Downloads/LJSpeech-1.1/metadata.csv'
out_path = os.environ['HOME'] + '/Downloads/LJSpeech-1.1/txt'

shutil.rmtree(out_path, ignore_errors=True)
os.mkdir(out_path)

abbreviations = {
    'mr': 'mister',
    'mrs.': 'misess',
    'dr.': 'doctor',
    'no.': 'number',
    'st.': 'saint',
    'co.': 'company',
    'jr.': 'junior',
    'maj.': 'major',
    'gen.': 'general',
    'drs.': 'doctors',
    'rev.': 'reverend',
    'lt.': 'lieutenant',
    'hon.': 'honorable',
    'sgt.': 'sergeant',
    'capt.': 'captain',
    'esq.': 'esquire',
    'ltd.': 'limited',
    'col.': 'colonel',
    'ft.': 'fort'
}

# Parse LJ csv file
with open(in_path, 'r') as input_file:
    for line in input_file:
        splitted_line = line.split('|')

        # Normalize text
        splitted_line[2] = splitted_line[2].lower()
        splitted_line[2] = ''.join(c for c in splitted_line[2] if c.isalnum() or c == ' ')

        # Expand abbreviations
        words = splitted_line[2].split()
        for i in range(len(words)):
            if words[i] in abbreviations:
                words[i] = abbreviations[words[i]]

        # Save to separate .txt file
        with open(out_path + '/' + splitted_line[0] + '.txt', 'w') as output_file:
            output_file.write(' '.join(words))
