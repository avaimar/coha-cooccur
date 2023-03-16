"""
Explores the frequency of surnames and otherization words in COHA.
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import json
from nltk.tag import pos_tag
import matplotlib
import pandas as pd
import seaborn as sns
from tqdm import tqdm

tqdm.pandas()


# Functions
def main(args):

    # Load word lists
    with open(f'{args.wlist_dir}/word_lists_all.json', 'r') as file:
        word_list_all = json.load(file)

    # Load PMI file: We can use #w in the PMI file to obtain the frequencies
    pmi = pd.read_csv(args.pmi_path)
    # Note: Because the word monstrous was repeated in the original Otherization word list,
    # we need to drop duplicates in the pmi DF. Duplicates also includes NAN for context.
    pmi.drop_duplicates(['w', 'c', 'word_list', 'decade'], inplace=True)

    # Otherization words
    wls = ['Otherization Words']
    word_df = pmi.loc[pmi['word_list'].isin(wls)].copy()

    # We drop duplicates because the PMI file rows are (w, c), and #w is the same for all these rows.
    word_df.drop_duplicates(subset=['w_idx', '#w', 'w', 'word_list', 'decade'], inplace=True)
    word_df.drop(['c_idx', '#wc', 'c', '#c'], axis=1, inplace=True)

    # Plot frequency
    for wl in wls:
        if wl != 'Otherization Words':
            continue
        wl_df = word_df.loc[word_df['word_list'] == wl].copy()

        cdict = {'aggressive': '#8a5bb1', 'brutal': '#ff7626', 'cruel': '#da2329', 'monstrous': '#009530'}
        wl_df['col'] = wl_df['w'].apply(lambda w: f'r{w}' if w in ['aggressive', 'brutal', 'cruel', 'monstrous'] else 'gray')
        plt.clf()
        ax = sns.lineplot(wl_df, x='decade', y='#w', units='w', hue='col', errorbar=None, legend=False, estimator=None)
        ax.set(ylabel='Decadal frequency', xlabel='Year')

        #sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=5, title=None, frameon=False)
        for w in ['aggressive', 'brutal', 'cruel', 'monstrous']:
            ax.text(
                1992,
                wl_df.loc[(wl_df['decade'] == 1990) & (wl_df['w'] == w)].iloc[0]['#w'],
                w, color=cdict[w])
        ax.set_xticks(range(1800, 2001, 20), labels=range(1800, 2001, 20))
        ax.figure.savefig(os.path.join(args.output_dir, 'otherization.png'), dpi=400)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-output_dir", type=str)
    parser.add_argument("-wlist_dir", type=str)
    parser.add_argument("-hw_input_dir", type=str)
    parser.add_argument("-pmi_path", type=str)

    args = parser.parse_args()

    # Paths
    args.output_dir = '../results/Frequencies'
    args.wlist_dir = '../../Local/word_lists/'
    args.coha_vecs = '../../Local/data/raw/coha'
    args.pmi_path = '../results/PMI/pmi.csv'
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)


