"""
Explore overlap between otherization and surnames.
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from nltk.tag import pos_tag
import matplotlib
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from bias_utils import load_SPPMI, load_coha

tqdm.pandas()

word_types_agg_1990_PNASWhite = {
    'names': [
       'anthony',   'babbitt', 'barnett', 'benjamin', 'calvert',
       'capitalism', 'cara', 'carl', 'carolina', 'delaney', 'delany', 'du',  'dunne', 'finley','fitch',
        'harris', 'howard', 'hussein', 'isaac', 'jacobson', 'jay', 'ken', 'leno',  'lexington', 'lincoln',
        'marc', 'mario',   'montgomery', 'phil', 'reginald', 'robertson', 'robinson', 'saddam',
       'sinclair', 'stanford', 'vivian', 'von',  'walter'],
}


def measure_overlap(m, wls, w1, w2, return_contexts=False, vocab=None):
    # Get word indices
    w1_tup = list(set([(m.key_to_index[w], w) for w in wls[w1] if w in m]))
    w2_tup = list(set([(m.key_to_index[w], w) for w in wls[w2] if w in m]))

    if len(w1_tup) == 0 or len(w2_tup) == 0:
        return pd.DataFrame()

    # Get word vectors
    w1_idx, w1_wds = list(zip(*w1_tup))
    w2_idx, w2_wds = list(zip(*w2_tup))

    w1_vecs = m.vectors[w1_idx, :]
    w2_vecs = m.vectors[w2_idx, :]

    if return_contexts:
        # Get non-zero elements
        w1_vecs_int = (w1_vecs > 1e-6).astype(int)
        w2_vecs_int = (w2_vecs > 1e-6).astype(int)

        # Get non-zero context words shared by surnames and attributes
        overlap = np.multiply(np.expand_dims(w1_vecs_int, axis=1), w2_vecs_int)
        overlap = np.transpose(overlap.nonzero())
        overlap = pd.DataFrame(overlap)
        overlap.rename(columns={0: 'surname', 1: 'index', 2: 'context'}, inplace=True)

        # Convert indices to words
        overlap['surname'] = overlap['surname'].apply(lambda w: w1_wds[w])
        overlap['index'] = overlap['index'].apply(lambda w: w2_wds[w])
        overlap['context'] = overlap['context'].apply(lambda w: vocab[w])

        return overlap[['index', 'surname', 'context']]

    overlap = np.matmul((w2_vecs > 1e-6).astype(int), (w1_vecs > 1e-6).astype(int).T)
    overlap = pd.DataFrame(overlap)
    overlap.rename(columns={i: w for i, w in enumerate(w1_wds)}, inplace=True)
    overlap.rename(index={i: w for i, w in enumerate(w2_wds)}, inplace=True)
    overlap.reset_index(drop=False, inplace=True)

    # Melt
    overlap = overlap.melt(
        id_vars=['index'], var_name='surname', value_name='2nd degree (shared context words)')

    return overlap


def SPPMI(vectors, w, c, decade):
    # Note that this computation is symmetric for w, c
    m = vectors[decade]
    w_idx = m.key_to_index[w]
    c_idx = m.key_to_index[c]
    return m.vectors[w_idx, c_idx]


def plot_heatmap(df, col, title, figname, group, col_wrap, share_cbar):

    if share_cbar:
        vmin, vmax = 0, df[col].max()
    else:
        vmin, vmax = None, None

    def facet_heatmap(data, color, **kws):
        data = data.pivot_table(values=col, index='index', columns='surname')
        sns.heatmap(data, cbar=True, vmin=vmin, vmax=vmax)

    #if col == 'Value':
        #plt.figure(figsize=(6, 3), dpi=900)
    g = sns.FacetGrid(df, col=group, col_wrap=col_wrap)
    g.map_dataframe(facet_heatmap)
    g.set_titles(row_template="{row_name}", col_template='{col_name}')
    g.fig.suptitle(title)
    g.figure.savefig(os.path.join(args.output_dir, figname),
                     dpi=800
                     )


def explore_surnames(v, wls, w1, vocab):
    top_words = ['aggressive', 'brutal', 'cruel', 'monstrous']
    context_df = pd.DataFrame()
    for decade, model in v.items():
        contexts = measure_overlap(
            m=model, wls=wls, w1=w1, w2='Otherization Words',
            return_contexts=True, vocab=vocab)
        contexts['decade'] = int(decade)
        context_df = pd.concat([context_df, contexts])

    # Filter for top words
    context_df = context_df.loc[context_df['index'].isin(top_words)]
    common_df = context_df.groupby(
        ['context', 'decade', 'index'])['surname'].count().reset_index()

    # View single decade
    c1990 = common_df.loc[common_df['decade'] == 1990]
    c1990 = c1990.sort_values(['index', 'surname'], ascending=False)
    c1990 = pd.pivot_table(c1990, index=['index'], values=['context', 'surname'], aggfunc=lambda x: list(x)).reset_index()

    if w1 == 'PNAS White Target Words':
        # Histogram
        aggressive_1990 = common_df.loc[(common_df['index'] == 'aggressive') & (common_df['decade'] == 1990)].copy()
        aggressive_1990['Context type'] = aggressive_1990['context'].apply(
            lambda w: 'Name' if w in word_types_agg_1990_PNASWhite['names'] else 'Other')

        plt.clf()
        ax = sns.barplot(
            aggressive_1990.sort_values('surname', ascending=False), x='context', y='surname', hue='Context type')
        #ax.tick_params(axis='x', rotation=90, labelsize=2)
        ax.set(xlabel='Context', ylabel='Number of surnames that share the context word', xticklabels=[])
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"surname_frequency_aggressive_{w1}"), dpi=1200)

    # Aggregate across decades
    agg_df = context_df.groupby(
        ['context', 'index'])['surname'].count().reset_index()
    agg_df = agg_df.sort_values(['index', 'surname'], ascending=False)
    agg_df = pd.pivot_table(agg_df, index=['index'], values=['context', 'surname'], aggfunc=lambda x: list(x)).reset_index()

    # Single word across decades
    aggressive = common_df.loc[common_df['index'] == 'aggressive']
    aggressive = aggressive.sort_values(['decade', 'surname'], ascending=False).groupby('decade').head(5)

    cruel = common_df.loc[common_df['index'] == 'cruel']
    cruel = cruel.sort_values(['decade', 'surname'], ascending=False).groupby('decade').head(5)


# Functions
def main(args):

    # Load word lists
    with open(f'{args.wlist_dir}/word_lists_all.json', 'r') as file:
        word_list_all = json.load(file)
    word_lists = [
            'Asian_San_Bruno_All', 'White_San_Bruno_All',
            'PNAS Asian Target Words', 'PNAS White Target Words']

    # Load HistWords vocabulary (to get the specific contexts)
    HWvectors = load_coha(input_dir=args.hw_input_dir)
    vocabulary = list(HWvectors['1990'].key_to_index.keys())

    # Load PMI file
    pmi = pd.read_csv(args.pmi_path)
    # Note: Because the word monstrous was repeated in the original Otherization word list,
    # we need to drop duplicates in the pmi DF. Duplicates also includes NAN for context.
    pmi.drop_duplicates(['w', 'c', 'word_list', 'decade'], inplace=True)

    # Load vectors
    for negative in range(5, 6, 5):
        vectors = load_SPPMI(
            input_dir=os.path.join(args.vector_dir, 'vectors'), negative=negative)

        # Measure overlap
        overlap_df = pd.DataFrame()
        for wlist in word_lists:
            for decade, model in vectors.items():
                overlap = measure_overlap(m=model, wls=word_list_all, w1=wlist, w2='Otherization Words')
                overlap['decade'] = int(decade)
                overlap['Word List'] = wlist
                overlap_df = pd.concat([overlap_df, overlap])

        # (SPPMI(w, c))
        overlap_df['SPPMI'] = overlap_df.apply(lambda row: SPPMI(
            vectors=vectors, w=row['index'], c=row['surname'], decade=str(row['decade'])), axis=1)

        # First degree associations
        overlap_df = overlap_df.merge(
            pmi[['w', 'c', 'word_list', 'decade', '#wc']], how='left', validate='many_to_one',
            left_on=['surname', 'index', 'Word List', 'decade'], right_on=['w', 'c', 'word_list', 'decade']
        )
        overlap_df.rename(columns={'#wc': '1st degree (shared contexts)'}, inplace=True)

        for wlist in word_lists:
            # Heatmaps
            wlist_df = overlap_df.loc[overlap_df['Word List'] == wlist].copy()
            plot_heatmap(df=wlist_df, col='2nd degree (shared context words)', title='',
                         figname=f'2nd_overlap_{wlist}_{negative}.png', group='decade', col_wrap=5, share_cbar=True)

            plot_heatmap(df=wlist_df, col='1st degree (shared contexts)', title='',
                         figname=f'1st_overlap_{wlist}_{negative}.png', group='decade', col_wrap=5, share_cbar=True)

            plot_heatmap(df=wlist_df, col='SPPMI', title='',
                         figname=f'SPPMI_overlap_{wlist}_{negative}.png', group='decade', col_wrap=5, share_cbar=True)

        # Aggregate heatmaps (across decades)
        agg_df = overlap_df.copy()
        agg_df = agg_df.groupby(['index', 'surname', 'Word List'])[[
            '1st degree (shared contexts)', '2nd degree (shared context words)']].sum().reset_index()
        for wlist in word_lists:
            wlist_df = agg_df.loc[agg_df['Word List'] == wlist].copy()
            wlist_df = wlist_df.melt(id_vars=['index', 'surname', 'Word List'], var_name='Measure', value_name='Value')
            sns.set(font_scale=0.6)

            plot_heatmap(df=wlist_df, col='Value', title='', figname=f'agg_{wlist}_{negative}.png',
                         group='Measure', col_wrap=2, share_cbar=False)

        # Explore White surnames 2nd degree overlap
        explore_surnames(v=vectors, w1='PNAS White Target Words')
        print('h')



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-output_dir", type=str)
    parser.add_argument("-vector_dir", type=str)
    parser.add_argument("-wlist_dir", type=str)
    parser.add_argument("-pmi_path", type=str)
    parser.add_argument("-hw_input_dir", type=str)

    args = parser.parse_args()

    # Paths
    args.output_dir = '../results/SPPMI/analysis'
    args.wlist_dir = '../../Local/word_lists/'
    args.vector_dir = '../results/SPPMI/'
    args.pmi_path = '../results/PMI/pmi.csv'
    args.hw_input_dir = '../../Replication-Garg-2018/data/coha-word'
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
