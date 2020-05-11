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
def create_output_file(wav_filename:str) -> None:
    # Load mfcc
    fts = np.fromfile(
        corpus_path + '/fts/' + wav_filename,
        dtype=float
    )
    fts = fts.reshape((number_of_feats, -1))

    # Load TextGrid
    name = wav_filename[:-4]
    tokens = np.array(
        text_parser.load_textgrid(
            corpus_path + '/textgrid/' + name + '.TextGrid'
        ),
        dtype=float
    )
    if fts.shape[1] >= tokens.shape[0]:
        tokens = np.concatenate([
            tokens,
            np.zeros(
                fts.shape[1] - tokens.shape[0],
                dtype=float
            )
        ])
    else:
        tokens = tokens[:fts.shape[1]]
    tokens = tokens.reshape((1, -1))
    assert fts.shape[1] == tokens.shape[1]
    
    # Save output data
    tokens.tofile(
        corpus_path + '/output/' + name + '.npy'
    )
    print(name, 'DONE')


if __name__ == '__main__':
    # Process wav files
    print('CREATING OUTPUT DATA...')
    shutil.rmtree(corpus_path + '/output', ignore_errors=True)
    os.mkdir(corpus_path + '/output')
    pool = mp.Pool(processes=7)
    for wav_filename in os.listdir(corpus_path + '/fts'):
        if wav_filename.endswith(".npy"):
            # create_output_file(wav_filename)
            pool.apply_async(
                create_output_file, args=(wav_filename,)
            )
    pool.close()
    pool.join()
    print('OUTPUT DATA CREATED')
