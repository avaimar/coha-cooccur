import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# Paths
output_path = '../results/PMI'

# Load bias scores
bias_scores = pd.read_csv('../../Replication-Garg-2018/results/coha_all_SB_bias.csv')
bias_scores = bias_scores.loc[bias_scores['Word'] == '--Overall--']
bias_scores = bias_scores.loc[bias_scores['Bias Comparison'].isin([
    'PNAS - Asian/White Otherization Bias', 'San Bruno - Asian/White Otherization Bias'])]
bias_scores['decade'] = bias_scores['Time Period'].apply(lambda period: int(period[:4]))

bias_scores = bias_scores.pivot(
    index=['Time Period', 'decade', 'Word'], columns=['Bias Comparison'],
    values='RND Bias Score').reset_index()

# Load word_df
word_df = pd.read_csv(os.path.join(output_path, 'word_df.csv'))

# Correlation with quality of attributes, T1 and T2
for word_group in word_df['word_list'].unique():
    wgroup = word_df.loc[word_df['word_list'] == word_group].copy()
    wgroup = wgroup.groupby('decade').agg({'%_deviation_optimum': 'mean'}).reset_index()
    wgroup = pd.merge(wgroup, bias_scores, on='decade', validate='one_to_one')

    if word_group == 'Otherization Words' or 'San_Bruno' in word_group:
        ax = sns.scatterplot(
            wgroup, x='%_deviation_optimum', y='San Bruno - Asian/White Otherization Bias',
            hue='decade'
        )
        ax.set(xlabel='Deviation from optimum (%)')
        ax.figure.savefig(os.path.join(output_path, 'bias_corr_{}_SB.png'.format(word_group)))
        plt.clf()

    if word_group == 'Otherization Words' or 'PNAS' in word_group:
        ax = sns.scatterplot(
            wgroup.loc[wgroup['decade'] > 1890], x='%_deviation_optimum',
            y='PNAS - Asian/White Otherization Bias', hue='decade')
        ax.set(xlabel='Deviation from optimum (%)')
        ax.figure.savefig(os.path.join(output_path, 'bias_corr_{}_PNAS.png'.format(word_group)))
        plt.clf()
