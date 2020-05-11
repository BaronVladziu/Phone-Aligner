#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import multiprocessing as mp
import numpy as np
import os
import python_speech_features
import scipy.io.wavfile
import scipy.signal
import shutil


# Settings
corpus_path = './LJSpeech-1.1'
sampling_frequency = 22050


def process_wav(wav_filename:str) -> None:
    # Load wav file
    wav = scipy.io.wavfile.read(
        corpus_path + '/wavs/' + wav_filename
    )
    wav = np.array(wav[1], dtype=float)

    # Compute MFCC
    fts = python_speech_features.mfcc(
        wav,
        samplerate=sampling_frequency,
        winstep=0.01,
        nfft=1024
    )

    # Save MFCC
    name = wav_filename[:-4]
    fts.tofile(
        corpus_path + '/fts/' + name + '.npy'
    )
    print(name, 'DONE')


if __name__ == '__main__':
    # Process wav files
    print('COMPUTING SPECTROGRAMS...')
    shutil.rmtree(corpus_path + '/fts', ignore_errors=True)
    os.mkdir(corpus_path + '/fts')
    pool = mp.Pool(processes=7)
    for wav_filename in os.listdir(corpus_path + '/wavs'):
        if wav_filename.endswith(".wav"):
            # process_wav(wav_filename)
            pool.apply_async(
                process_wav, args=(wav_filename,)
            )
    pool.close()
    pool.join()
    print('SPECTROGRAMS COMPUTED')
