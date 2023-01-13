import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import src.bias_utils as bias_utils

# Params
# The Garg et al. computation does not consider whether otherization vectors are non-zero in a model.
match_garg = True

# Paths
output_path = '../results/PMI'

# Load word_df and vectors
word_df = pd.read_csv(os.path.join(output_path, 'word_df_median.csv'))

vectors = bias_utils.load_coha()

# Load word lists
with open('../../Local/word_lists/word_lists_all.json', 'r') as file:
    word_list_all = json.load(file)

# Generate bias scores by quartile
wls = ['Asian_San_Bruno_All', 'White_San_Bruno_All', 'PNAS Asian Target Words', 'PNAS White Target Words']
main_df = pd.DataFrame()
for decade, model in vectors.items():
    # Define otherization vectors
    otherization_idx = [model.key_to_index[w] for w in word_list_all['Otherization Words'] if
                        bias_utils.present_word(model, w)]
    if match_garg:
        otherization_idx = [model.key_to_index[w] for w in word_list_all['Otherization Words'] if w in model]
    otherization_vecs = model.vectors[otherization_idx, :]

    for wl in wls:
        # Generate %Dev quartiles
        decade_df = word_df.loc[(word_df['decade'] == int(decade)) & (
                word_df['Word List'] == wl.replace('_', ' '))].copy()
        try:
            decade_df['quartile'] = pd.qcut(
                decade_df['%_deviation_optimum'], q=4, labels=[0, 1, 2, 3])
        except (IndexError, ValueError):
            continue

        # Compute bias score for words in each $Dev group
        for quartile in range(4):
            surnames_dev_quart = list(decade_df.loc[decade_df['quartile'] == quartile]['w'].unique())
            if 'Asian' in wl:
                asian_mean_vec = bias_utils.compute_mean_vector(model, surnames_dev_quart)
                white_mean_vec = bias_utils.compute_mean_vector(model, word_list_all[wl.replace('Asian', 'White')])
            elif 'White' in wl:
                asian_mean_vec = bias_utils.compute_mean_vector(model, word_list_all[wl.replace('White', 'Asian')])
                white_mean_vec = bias_utils.compute_mean_vector(model, surnames_dev_quart)
            else:
                raise Exception('[ERROR] Check word list type.')

            if asian_mean_vec is not None and white_mean_vec is not None:
                bias_score, _, _ = bias_utils.compute_bias_score(
                    attribute_vecs=otherization_vecs, t1_mean=white_mean_vec, t2_mean=asian_mean_vec)

                # Append to main df
                quart_dict = {
                    'decade': [int(decade)], 'wl': [wl], '%Deviation (quartile)': [quartile], 'Bias score': [bias_score]
                }
                main_df = pd.concat([main_df, pd.DataFrame.from_dict(quart_dict)])


# Plot
main_df['wl'] = main_df['wl'].apply(lambda wl: wl.replace('_', ' '))

g = sns.FacetGrid(
    main_df.loc[main_df['decade'] > 1890], col="wl", col_wrap=2, margin_titles=True, legend_out=True,
    hue='decade', palette='rocket_r'
    )
g.set_axis_labels(x_var='Deviation from optimum (%) quartile')
g.set_titles(col_template="{col_name}")
g.map(sns.lineplot, '%Deviation (quartile)', 'Bias score')
g.add_legend()
g.figure.savefig(os.path.join(output_path, 'bias_corr_quartiles.png'))
