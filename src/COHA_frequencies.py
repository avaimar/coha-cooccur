"""
Generates a
"""

import argparse
import glob
from matrixserializer import load_matrix
import numpy as np
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from ioutils import load_pickle, mkdir


from PMI_compute import idx_to_word
from bias_utils import load_coha


def get_word_freq(w_dict, m, w=None, w_idx=None):
    if w_idx is None:
        assert w is not None
        try:
            w_idx = w_dict[w]
        except KeyError:
            return None

    try:
        w_coo = m[w_idx].tocoo()
    except IndexError:
        return None

    return np.sum(w_coo)


def get_frequencies(m, w_dict, w_set):
    # CSR matrix is in the format (word, context)
    m_df = pd.DataFrame()
    for w in w_set:
        w_sum = get_word_freq(w=w, w_dict=w_dict, m=m)
        if w_sum is None:
            continue

        w_df = pd.DataFrame.from_dict({'w': [w], '#w': [w_sum]})
        m_df = pd.concat([m_df, w_df])

    return m_df


def main(args):
    # Load word dictionary and word lists
    word_dict = load_pickle(args.word_dict_pkl)

    # Load HistWords vocabulary
    vectors = load_coha(input_dir=args.hwinput_dir)
    HW_vocab = list(vectors['1990'].key_to_index.keys())

    # Process each decadal matrix
    bin_files = glob.glob(os.path.join(args.bin_dir, '*.bin'))

    freq_df = pd.DataFrame()
    if os.path.exists(os.path.join(args.output_dir, 'coha_frequencies.csv')):
        freq_df = pd.read_csv(os.path.join(args.output_dir, 'coha_frequencies.csv'))

    for bin_file in tqdm(bin_files):
        decade = bin_file.split(os.path.sep)[-1].replace('.bin', '')

        if 'decade' in freq_df.columns and int(decade) in list(freq_df['decade'].unique()):
            continue
        matrix = load_matrix(bin_file)
        matrix_df = get_frequencies(m=matrix, w_dict=word_dict, w_set=HW_vocab)
        matrix_df['decade'] = int(decade)

        freq_df = pd.concat([freq_df, matrix_df])

        # Save output
        freq_df.to_csv(os.path.join(args.output_dir, 'coha_frequencies.csv'), index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-bin_dir", type=str)
    parser.add_argument("-word_dict_pkl", type=str)
    parser.add_argument("-output_dir", type=str)
    parser.add_argument("-hwinput_dir", type=str)

    args = parser.parse_args()

    # Paths
    args.bin_dir = '../cooccurs/word/4'
    args.word_dict_pkl = '../info/word-dict.pkl'
    args.hwinput_dir = '../../Replication-Garg-2018/data/coha-word'

    args.output_dir = os.path.join('..', 'results', f'Frequencies')
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
