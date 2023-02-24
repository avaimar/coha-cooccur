"""
Builds the SPPMI vectors. Note these only depend on the COHA corpus, and k.
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import glob
from gensim.models import KeyedVectors
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from bias_utils import load_coha, load_SPPMI, compute_bias_score, compute_mean_vector, present_word

tqdm.pandas()


def build_sppmi(df, vocab):
    # Generates the SPPMI vector for a single word given a vocabulary
    assert len(df['w_idx'].unique()) == 1

    # Drop duplicates and missing context values
    df.drop_duplicates(['w_idx', 'c_idx'], inplace=True)
    df = df.loc[~df['c'].isna()]

    sppmi = pd.DataFrame(vocab, columns=['key'])
    try:
        sppmi = sppmi.merge(
            df[['c', 'SPPMI']], left_on='key', right_on='c', how='left', validate='one_to_one')
    except pd.errors.MergeError:
        print(df['w_idx'].unique())
    # We fill missings with nan since PMI(w, c) for a missing entry is -inf, so SPPMI is 0.
    sppmi.fillna(0, inplace=True)
    return sppmi['SPPMI']


def build_sppmi_vectors(args):
    # Load PMI data
    pmi_df = pd.read_csv(os.path.join(args.output_dir, 'PMI', 'pmi_eq5.csv'))

    # Build SPPMI rows for each set of names. Note we use only one dimensionality as SPPMI only
    # varies with k, not d. The only column varying with d is w_dot_c, which is not used for SPPMI.
    pmi_df = pmi_df.loc[pmi_df['d'] == 300].copy()

    # Use HistWords for fixed vocabulary / fixed word vectors
    vectors = load_coha(input_dir=args.input_dir)
    vocab = list(vectors['1990'].key_to_index.keys())

    for k in tqdm(pmi_df['k'].unique()):
        for decade in pmi_df['decade'].unique():
            k_d_file = os.path.join(args.results_dir, 'vectors', f'sppmi-{k}-{decade}.kv')
            if os.path.exists(k_d_file):
                continue
            pmi_k_dec = pmi_df.loc[(pmi_df['k'] == k) & (pmi_df['decade'] == decade)].copy()

            # Need to drop duplicates (some words belong to multiple WLs)
            pmi_k_dec.drop_duplicates(subset=['w_idx', 'c_idx'], inplace=True)

            # Create array
            sppmi_vecs = np.zeros((len(vocab), len(pmi_k_dec['w_idx'].unique())))
            sppmi_words = []

            for i, w_idx in enumerate(pmi_k_dec['w_idx'].unique()):
                w_df = pmi_k_dec.loc[pmi_k_dec['w_idx'] == w_idx].copy()
                w_sppmi = build_sppmi(w_df, vocab)
                sppmi_vecs[:, i] = w_sppmi
                sppmi_words.append(list(w_df['w'].unique())[0])

            # Create gensim vectors and save
            model = KeyedVectors(len(vocab))
            model.add_vectors(keys=sppmi_words, weights=sppmi_vecs.T)
            os.makedirs(os.path.join(args.output_dir, '..', 'SPPMI'), exist_ok=True)
            model.save(k_d_file)


def bias_scores(wls):
    if os.path.exists(f'{args.results_dir}/bias/bias_scores.csv'):
        bias_df = pd.read_csv(f'{args.results_dir}/bias/bias_scores.csv')
        return bias_df

    bias_df = pd.DataFrame()
    for negative in range(5, 30, 5):
        vectors = load_SPPMI(
            input_dir=os.path.join(args.results_dir, 'vectors'), negative=negative)

        for decade, model in vectors.items():
            for source_list in ['San Bruno', 'PNAS']:
                # Get lists
                asian_surnames = wls['Asian_San_Bruno_All'] if source_list == 'San Bruno' else wls['PNAS Asian Target Words']
                white_surnames = wls['White_San_Bruno_All'] if source_list == 'San Bruno' else wls['PNAS White Target Words']

                # Get mean vectors (note: these are post normalized)
                asian_vec = compute_mean_vector(model=model, words=asian_surnames, pre_normalize=True)
                white_vec = compute_mean_vector(model=model, words=white_surnames, pre_normalize=True)

                # Get attribute vecs (need to be normalized)
                attribute_vecs = [model.key_to_index[w] for w in list(set(wls['Otherization Words'])) if present_word(model, w)]
                attribute_vecs = model.vectors[attribute_vecs, :]
                attribute_vecs = attribute_vecs / np.linalg.norm(attribute_vecs, axis=1).reshape(-1, 1)

                # Compute "active" surnames
                n_Asian = len([w for w in asian_surnames if present_word(model, w)])
                n_White = len([w for w in white_surnames if present_word(model, w)])
                n_other = len([w for w in set(wls['Otherization Words']) if present_word(model, w)])

                # Compute bias
                if asian_vec is not None and white_vec is not None:
                    bscore, _, _ = compute_bias_score(
                        attribute_vecs=attribute_vecs, t1_mean=white_vec, t2_mean=asian_vec, cosine=True)

                    df = pd.DataFrame.from_dict(
                        {'Word List': [source_list], 'Bias score': [bscore], 'decade': [decade], 'k': [negative],
                         'n_Asian': [n_Asian], 'n_White': [n_White], 'n_otherization': [n_other]})
                    bias_df = pd.concat([bias_df, df])

    # Save file
    os.makedirs(f'{args.results_dir}/bias', exist_ok=True)
    bias_df.to_csv(f'{args.results_dir}/bias/bias_scores.csv', index=False)
    return bias_df


def plot_bias(df):
    sns.set_theme(style="white", font_scale=1.8)
    g = sns.FacetGrid(
        df, row="Word List", margin_titles=True, legend_out=True,
        hue='Word List', height=5.5, aspect=3.5)
    for ax in g.axes.flatten():
        ax.axhline(0, color='grey')
    g.map(sns.lineplot, 'decade', 'Bias score', errorbar=None)
    g.set_axis_labels(x_var='Decade', y_var='Bias score (cosine)')
    g.set_titles(row_template="{row_name}")
    g.set(xticks=np.arange(1810, 2001, 10))
    g.figure.savefig(f'{args.results_dir}/bias/bias_trends.png')

    # Surname frequency
    freq = df.copy()
    freq = freq.melt(
        id_vars=['decade', 'Word List'], value_vars=['n_Asian', 'n_White'],
        var_name='Surname', value_name='n')

    sns.set_theme(style="white", font_scale=1.8)
    g = sns.FacetGrid(
        freq, row="Word List", margin_titles=True, legend_out=True,
        hue='Surname', height=5.5, aspect=3.5, sharey=False)
    for ax in g.axes.flatten():
        ax.axhline(0, color='grey')
    g.map(sns.lineplot, 'decade', 'n')
    g.set_axis_labels(x_var='Decade', y_var='Number of unique names')
    g.set_titles(row_template="{row_name}")
    g.set(xticks=np.arange(1810, 2001, 10))
    plt.legend(title='Surname')
    g.figure.savefig(f'{args.results_dir}/bias/name_freq.png')


# Functions
def main(args):
    # Build vectors
    print('[INFO] Building SPPMI vectors')
    build_sppmi_vectors(args)

    # Load word lists
    with open(f'{args.wlist_dir}/word_lists_all.json', 'r') as file:
        word_list_all = json.load(file)

    # Compute bias scores
    print('[INFO] Computing bias scores')
    bias_df = bias_scores(wls=word_list_all)
    plot_bias(df=bias_df)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-output_dir", type=str)
    parser.add_argument("-input_dir", type=str)
    parser.add_argument("-results_dir", type=str)
    parser.add_argument("-wlist_dir", type=str)

    args = parser.parse_args()

    # Paths
    args.output_dir = '../results/SGNS/'
    args.input_dir = '../../Replication-Garg-2018/data/coha-word'
    args.wlist_dir = '../../Local/word_lists/'
    args.results_dir = '../results/SPPMI/'
    os.makedirs(args.results_dir, exist_ok=True)

    main(args)
