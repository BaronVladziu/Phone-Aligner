#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import multiprocessing as mp
import numpy as np
import os
import shutil

from phones import TextParser


# Settings
corpus_path = './LJSpeech-1.1'
sampling_frequency = 22050
number_of_feats = 13


text_parser = TextParser()
def create_input_file(wav_filename:str) -> None:
    # Load mfcc
    fts = np.fromfile(
        corpus_path + '/fts/' + wav_filename,
        dtype=float
    )
    fts = fts.reshape((number_of_feats, -1))

    # Load text
    name = wav_filename[:-4]
    txt_file = open(
        corpus_path + '/txt/' + name + '.txt',
        "r"
    )
    text = txt_file.readlines()[0]
    
    # Transform text to phone tokens
    tokens = np.array(
        text_parser.txt2tkn(text),
        dtype=float
    )
    tokens = np.concatenate([
        tokens,
        np.zeros(fts.shape[1] - tokens.shape[0], dtype=float)
    ])
    tokens = tokens.reshape((1, -1))

    # Merge feats and tokens
    input_data = np.concatenate([
        fts,
        tokens
    ])

    # Save input data
    input_data.tofile(
        corpus_path + '/input/' + name + '.npy'
    )
    print(name, 'DONE')
    
    with open('tmp.txt', 'a') as tmp:
        tmp.write(str(input_data.shape[1]) + '\n')



if __name__ == '__main__':
    # Process wav files
    print('CREATING INPUT DATA...')
    shutil.rmtree(corpus_path + '/input', ignore_errors=True)
    os.mkdir(corpus_path + '/input')
    pool = mp.Pool(processes=7)
    for wav_filename in os.listdir(corpus_path + '/fts'):
        if wav_filename.endswith(".npy"):
            # try:
            #     create_input_file(wav_filename)
            # except KeyError as e:
            #     print(wav_filename, str(e))

            pool.apply_async(
                create_input_file, args=(wav_filename,)
            )
    pool.close()
    pool.join()
    print('INPUT DATA CREATED')
