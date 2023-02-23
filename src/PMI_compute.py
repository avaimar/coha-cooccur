import argparse
import glob
import json
from matrixserializer import load_matrix
from nltk.corpus import stopwords
import numpy as np
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from ioutils import load_pickle, mkdir

from bias_utils import load_coha, load_coha_SGNS


def get_word_df(w_dict, m, w=None, w_idx=None):
    if w_idx is None:
        assert w is not None
        try:
            w_idx = w_dict[w]
        except KeyError:
            return None, None

    try:
        w_coo = m[w_idx].tocoo()
        w_df = pd.DataFrame({'w_idx': w_coo.row, 'c_idx': w_coo.col, '#wc': w_coo.data})
    except IndexError:
        return None, None

    w_sum = w_df['#wc'].sum()
    return w_df, w_sum


def dot_product(w, c, vec):
    try:
        w_idx_v = vec.key_to_index[w]
        c_idx_v = vec.key_to_index[c]
    except KeyError:
        return None

    v_w = vec.vectors[w_idx_v]
    v_c = vec.vectors[c_idx_v]

    if np.linalg.norm(v_w) < 1e-6 or np.linalg.norm(v_c) < 1e-6:
        return None

    return np.dot(v_w, v_c)


def idx_to_word(idx, w_dict):
    w = list(w_dict.keys())[idx]
    assert idx == w_dict[w]
    return w


# Get surname metrics for each decade
# CSR matrix is in the format (word, context)
def process_matrix(m, w_dict, word_lists, v):
    wls = ['Asian_San_Bruno_All', 'White_San_Bruno_All', 'Otherization Words',
           'PNAS Asian Target Words', 'PNAS White Target Words']
    m_df = pd.DataFrame()
    for wl in tqdm(wls):
        for w in word_lists[wl]:
            w_df, w_sum = get_word_df(w=w, w_dict=w_dict, m=m)
            if w_df is None:
                continue

            # Get #(w,c), #(w), #(c)
            w_df['w'] = w
            w_df['w_idx'] = w_df['w'].apply(lambda word: w_dict[word])
            w_df['c'] = w_df['c_idx'].apply(lambda idx: idx_to_word(idx, w_dict))
            w_df['#w'] = w_sum
            w_df['#c'] = w_df['c_idx'].apply(lambda idx: get_word_df(w_idx=idx, w_dict=w_dict, m=m)[1])

            # Get w.c (note: we check that ||w||2 and ||c||2 >= 1e-6)
            w_df['w_dot_c'] = w_df.apply(
                lambda row: dot_product(w=row['w'], c=row['c'], vec=v), axis=1)

            # Append
            w_df['word_list'] = wl
            m_df = pd.concat([m_df, w_df])

    # Add |D|
    m_df['D'] = m.sum()

    return m_df


def main(args, vectors):
    # Load word dictionary and word lists
    word_dict = load_pickle(args.word_dict_pkl)
    with open(f'{args.wlist_dir}/word_lists_all.json', 'r') as file:
        word_list_all = json.load(file)

    # Add stopwords
    word_list_all['Stopwords'] = stopwords.words('english')

    # Process each decadal matrix
    bin_files = glob.glob(os.path.join(args.bin_dir, '*.bin'))

    pmi_df = pd.DataFrame()
    if os.path.exists(os.path.join(args.output_dir, 'pmi.csv')):
        pmi_df = pd.read_csv(os.path.join(args.output_dir, 'pmi.csv'))

    for bin_file in tqdm(bin_files):
        decade = bin_file.split(os.path.sep)[-1].replace('.bin', '')

        if 'decade' in pmi_df.columns and int(decade) in list(pmi_df['decade'].unique()):
            continue
        #if decade != '1990':
        #    continue
        matrix = load_matrix(bin_file)
        matrix_df = process_matrix(m=matrix, w_dict=word_dict, word_lists=word_list_all, v=vectors[decade])
        matrix_df['decade'] = int(decade)

        # Compute PMI
        matrix_df['PMI'] = matrix_df.apply(
            lambda row: np.log((row['#wc'] * row['D']) / (row['#w'] * row['#c'])) - np.log(args.K) if row['#w'] * row[
                '#c'] != 0 else None, axis=1)

        pmi_df = pd.concat([pmi_df, matrix_df])

        # Save output
        pmi_df.to_csv(os.path.join(args.output_dir, 'pmi.csv'), index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-K", type=int)
    parser.add_argument("-d", type=int, default=300)
    parser.add_argument("-vectors", type=str)
    parser.add_argument("-vectors_dir", type=str)
    parser.add_argument("-wlist_dir", type=str)
    parser.add_argument("-bin_dir", type=str)
    parser.add_argument("-word_dict_pkl", type=str)
    parser.add_argument("-output_dir", type=str)

    args = parser.parse_args()

    # Paths
    args.bin_dir = '../cooccurs/word/4'
    args.wlist_dir = '../../Local/word_lists/'
    if args.vectors == 'HistWords':
        args.vectors_dir = '../../Replication-Garg-2018/data/coha-word'
    else:
        args.vectors_dir = '../../COHA-SGNS/results/vectors'
    args.word_dict_pkl = '../info/word-dict.pkl'

    if args.vectors == 'HistWords':
        args.K = 5  # HistWords indicate they follow Levy et al. (2015) in order to set hyperparameters.
        # Seems like k is 5 per https://aclanthology.org/Q15-1016.pdf but unsure
        assert args.d == 300

        args.output_dir = os.path.join('..', 'results', f'{args.vectors}', 'PMI')
        os.makedirs(args.output_dir)

        vectors = load_coha(args.vectors_dir)
        main(args, vectors)

    elif args.vectors == 'SGNS':
        # Need to run for each negative sampling parameter
        for negative in tqdm(list(range(0, 30, 5))):
            args.output_dir = os.path.join('..', 'results', 'SGNS', f'{args.vectors}-{negative}-{args.d}', 'PMI')
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

            args.K = negative
            vectors = load_coha_SGNS(input_dir=args.vectors_dir, negative=negative, d=args.d, norm=True)
            main(args, vectors)
    else:
        raise Exception('[ERROR] Check vectors. ')
