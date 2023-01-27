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
regression_df = pd.DataFrame()
for decade, model in tqdm(vectors.items()):
    # Define otherization vectors
    otherization_idx = [model.key_to_index[w] for w in word_list_all['Otherization Words'] if
                        bias_utils.present_word(model, w)]
    if match_garg:
        otherization_idx = [model.key_to_index[w] for w in word_list_all['Otherization Words'] if w in model]
    otherization_vecs = model.vectors[otherization_idx, :]

    # Absolute quartiles (quartiles across all word lists in a decade)
    decade_df = word_df.loc[word_df['decade'] == int(decade)].copy()
    try:
        decade_df['absolute_quartile'] = pd.qcut(
            decade_df['%_deviation_optimum'], q=4, labels=[0, 1, 2, 3])
    except (IndexError, ValueError):
        continue

    # Generate %Dev relative quartiles (quartiles across the word list in a decade)
    try:
        decade_df['relative_quartile'] = decade_df.groupby('Word List')['%_deviation_optimum'].transform(
            lambda x: pd.qcut(x, q=4, labels=[0, 1, 2, 3]))
    except (IndexError, ValueError):
        continue

    # Ex 1
    for wl in wls:
        wl_df = decade_df.loc[decade_df['Word List'] == wl.replace('_', ' ')].copy()

        # Ex 1: Compute bias score for words in each %Dev group: FIX other target group, relative quartile
        # for main Target group.
        for quartile in range(4):
            surnames_dev_quart = list(wl_df.loc[wl_df['relative_quartile'] == quartile]['w'].unique())
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
                    'Experiment': ['relative-fixed'], 'decade': [int(decade)], 'wl': [wl],
                    '%Deviation (quartile)': [quartile], 'Bias score': [bias_score]
                }
                main_df = pd.concat([main_df, pd.DataFrame.from_dict(quart_dict)])

    # ** Ex 2, 3: Cartesian product: relative-relative and absolute-absolute
    for quartile_type in ['relative', 'absolute']:
        for wl in ['{}_San_Bruno_All', 'PNAS {} Target Words']:

            asian_df = decade_df.loc[decade_df['Word List'] == wl.format('Asian').replace('_', ' ')].copy()
            white_df = decade_df.loc[decade_df['Word List'] == wl.format('White').replace('_', ' ')].copy()

            for quartile_i in range(4):
                for quartile_j in range(4):
                    asian_surnames_dev_quart = list(
                        asian_df.loc[asian_df['{}_quartile'.format(quartile_type)] == quartile_i]['w'].unique())
                    white_surnames_dev_quart = list(
                        white_df.loc[white_df['{}_quartile'.format(quartile_type)] == quartile_j]['w'].unique())

                    asian_mean_vec = bias_utils.compute_mean_vector(model, asian_surnames_dev_quart)
                    white_mean_vec = bias_utils.compute_mean_vector(model, white_surnames_dev_quart)

                    if asian_mean_vec is not None and white_mean_vec is not None:
                        bias_score, _, _ = bias_utils.compute_bias_score(
                            attribute_vecs=otherization_vecs, t1_mean=white_mean_vec, t2_mean=asian_mean_vec)

                        # Append to main df
                        quart_dict = {
                            'Experiment': ['{}-{}'.format(quartile_type, quartile_type)], 'decade': [int(decade)],
                            'wl': ['{}'.format(wl)],
                            '%Deviation (quartile)': ['{}-{}'.format(quartile_i, quartile_j)], 'Bias score': [bias_score]
                        }
                        main_df = pd.concat([main_df, pd.DataFrame.from_dict(quart_dict)])

    # ** Ex 4: Varying otherization words
    for wl in ['{}_San_Bruno_All', 'PNAS {} Target Words']:

        asian_df = decade_df.loc[decade_df['Word List'] == wl.format('Asian').replace('_', ' ')].copy()
        white_df = decade_df.loc[decade_df['Word List'] == wl.format('White').replace('_', ' ')].copy()
        otherization_df = decade_df.loc[decade_df['Word List'] == 'Otherization Words'].copy()

        for quartile_a in range(4):
            for quartile_w in range(4):
                for quartile_o in range(4):
                    # Define surnames
                    asian_surnames_dev_quart = list(
                        asian_df.loc[asian_df['relative_quartile'] == quartile_a]['w'].unique())
                    white_surnames_dev_quart = list(
                        white_df.loc[white_df['relative_quartile'] == quartile_w]['w'].unique())

                    asian_mean_vec = bias_utils.compute_mean_vector(model, asian_surnames_dev_quart)
                    white_mean_vec = bias_utils.compute_mean_vector(model, white_surnames_dev_quart)

                    # Define otherization words
                    otherization_dev_quart = list(
                        otherization_df.loc[otherization_df['relative_quartile'] == quartile_o]['w'].unique())
                    otherization_idx = [model.key_to_index[w] for w in otherization_dev_quart if
                                        bias_utils.present_word(model, w)]
                    if match_garg:
                        otherization_idx = [model.key_to_index[w] for w in otherization_dev_quart if
                                            w in model]
                    otherization_vecs = model.vectors[otherization_idx, :]

                    if asian_mean_vec is not None and white_mean_vec is not None and otherization_vecs is not None:
                        bias_score, _, _ = bias_utils.compute_bias_score(
                            attribute_vecs=otherization_vecs, t1_mean=white_mean_vec, t2_mean=asian_mean_vec)

                        # Append to main df (Exp 4)
                        quart_dict = {
                            'Experiment': ['otherization'], 'decade': [int(decade)],
                            'wl': ['{}'.format(wl)],
                            '%Deviation (quartile)': ['{}-{}-{}'.format(quartile_a, quartile_w, quartile_o)],
                            'Bias score': [bias_score]
                        }
                        main_df = pd.concat([main_df, pd.DataFrame.from_dict(quart_dict)])

    # Exp 5:
    for wl in ['{}_San_Bruno_All', 'PNAS {} Target Words']:

        asian_df = decade_df.loc[decade_df['Word List'] == wl.format('Asian').replace('_', ' ')].copy()
        white_df = decade_df.loc[decade_df['Word List'] == wl.format('White').replace('_', ' ')].copy()
        otherization_df = decade_df.loc[decade_df['Word List'] == 'Otherization Words'].copy()

        for quartile in range(5):
            # Define surnames
            if quartile == 4:  # We grab all surnames
                asian_quart_df, white_quart_df = asian_df.copy(), white_df.copy()
            else:
                asian_quart_df = asian_df.loc[asian_df['relative_quartile'] == quartile].copy()
                white_quart_df = white_df.loc[white_df['relative_quartile'] == quartile].copy()
            asian_surnames_dev_quart = list(asian_quart_df['w'].unique())
            white_surnames_dev_quart = list(white_quart_df['w'].unique())

            asian_mean_vec = bias_utils.compute_mean_vector(model, asian_surnames_dev_quart)
            white_mean_vec = bias_utils.compute_mean_vector(model, white_surnames_dev_quart)

            for other_w in otherization_df['w']:
                # Define otherization words
                otherization_idx = [model.key_to_index[w] for w in [other_w] if
                                    bias_utils.present_word(model, w)]
                if match_garg:
                    otherization_idx = [model.key_to_index[w] for w in [other_w] if w in model]
                otherization_vecs = model.vectors[otherization_idx, :]

                if asian_mean_vec is not None and white_mean_vec is not None and otherization_vecs.shape[0] != 0:
                    bias_score, t1_component, t2_component = bias_utils.compute_bias_score(
                        attribute_vecs=otherization_vecs, t1_mean=white_mean_vec, t2_mean=asian_mean_vec)

                    # Otherization word dict
                    other_w_dict = otherization_df.loc[otherization_df['w'] == other_w]

                    # Append to main df (Exp 5)
                    quart_dict = {
                        'Experiment': ['regression'], 'decade': [int(decade)], 'wl': ['{}'.format(wl)],
                        'other_word': [other_w],
                        'Norm': [t1_component.item()], 'Group': ['White'],
                        '%Deviation Target (quartile)': [quartile],
                        '%Deviation Target': [white_quart_df['%_deviation_optimum'].mean()],
                        '%Deviation Otherization': [other_w_dict['%_deviation_optimum'].iloc[0]],
                        '%Deviation Otherization (quartile)': [other_w_dict['relative_quartile'].iloc[0]],
                        'bias_score': [bias_score]

                    }
                    regression_df = pd.concat([regression_df, pd.DataFrame.from_dict(quart_dict)])

                    quart_dict = {
                        'Experiment': ['regression'], 'decade': [int(decade)], 'wl': ['{}'.format(wl)],
                        'other_word': [other_w],
                        'Norm': [t2_component.item()], 'Group': ['Asian'],
                        '%Deviation Target (quartile)': [quartile],
                        '%Deviation Target': [asian_quart_df['%_deviation_optimum'].mean()],
                        '%Deviation Otherization': [other_w_dict['%_deviation_optimum'].iloc[0]],
                        '%Deviation Otherization (quartile)': [other_w_dict['relative_quartile'].iloc[0]],
                        'bias_score': [bias_score]
                    }
                    regression_df = pd.concat([regression_df, pd.DataFrame.from_dict(quart_dict)])


main_df['wl'] = main_df['wl'].apply(lambda wl: wl.replace('_', ' '))
main_df.to_csv(os.path.join(output_path, 'quartile_df_MG{}.csv'.format(match_garg)), index=False)

regression_df['wl'] = regression_df['wl'].apply(lambda wl: wl.replace('_', ' '))
regression_df.to_csv(os.path.join(output_path, 'regression_df_MG{}.csv'.format(match_garg)), index=False)


# Load in
main_df = pd.read_csv(os.path.join(output_path, 'quartile_df.csv'))

# Plot Ex 1
exp1 = main_df.loc[(main_df['Experiment'] == 'relative-fixed') & (main_df['decade'] > 1890)].copy()
g = sns.FacetGrid(
    exp1, col="wl", col_wrap=2, margin_titles=True, legend_out=True,
    hue='decade', palette='rocket_r')
g.set_axis_labels(x_var='Deviation from optimum (%) quartile')
g.set_titles(col_template="{col_name}")
g.map(sns.lineplot, '%Deviation (quartile)', 'Bias score')
g.add_legend()
g.figure.savefig(os.path.join(output_path, 'Exp1_bias_corr_quartiles.png'))

# Plot Ex 2, 3
for quartile_type in ['relative', 'absolute']:
    exp = main_df.loc[
        (main_df['Experiment'] == '{}-{}'.format(quartile_type, quartile_type)) & (main_df['decade'] > 1890)].copy()
    exp[['%Dev quartile (Asian)', '%Dev quartile (White)']] = exp['%Deviation (quartile)'].apply(
        lambda x: pd.Series(x.split('-')))
    exp['xaxis'] = 0

    for wl in ['{}_San_Bruno_All', 'PNAS {} Target Words']:
        g = sns.FacetGrid(
            exp.loc[exp['wl'] == wl], col="%Dev quartile (White)", row='%Dev quartile (Asian)',
            margin_titles=True, legend_out=True, hue='decade', palette='rocket_r')
        g.set_axis_labels('')
        g.set_titles(col_template="White quartile {col_name}", row_template="Asian quartile {row_name}")
        #g.set_titles(col_template="", row_template="")
        g.map(plt.axhline, y=0, ls='--', c='gray', alpha=0.3, linewidth=0.5)
        g.map(sns.scatterplot, 'xaxis', 'Bias score')
        g.add_legend()

        for ax in g.axes.flatten():
            ax.set_xlabel("")

        g.set(xticks=[])
        g.figure.savefig(os.path.join(output_path, 'Exp_{}_{}_bias_corr_quartiles.png'.format(quartile_type, wl)))

# Plot Ex 4: Otherization
exp4 = main_df.loc[(main_df['Experiment'] == 'otherization') & (main_df['decade'] > 1890)].copy()
# * Collapse decades
exp4 = exp4.groupby(['wl', '%Deviation (quartile)'])['Bias score'].mean().reset_index()
exp4[['%Dev quartile (Asian)', '%Dev quartile (White)', '%Dev quartile (Otherization)']] = exp4['%Deviation (quartile)'].apply(
        lambda x: pd.Series(x.split('-')))

for wl in exp4['wl'].unique():
    exp4_wl = exp4.loc[exp4['wl'] == wl].copy()
    g = sns.FacetGrid(
        exp4_wl, col="%Dev quartile (Otherization)", margin_titles=True, legend_out=True,
        hue='%Dev quartile (White)', palette='rocket_r')
    g.set_axis_labels(x_var='Deviation from optimum (%) quartile')
    g.set_titles(col_template="{col_name}")
    g.map(sns.lineplot, '%Dev quartile (Asian)', 'Bias score')
    g.add_legend()
    g.figure.savefig(os.path.join(output_path, 'Exp4_{}.png'.format(wl.replace('{}', '').replace(' ', ''))))

# Explore PNAS vs SBruno
surname_df = pd.DataFrame()
for decade, model in vectors.items():

    # Absolute quartiles (quartiles across all word lists in a decade)
    decade_df = word_df.loc[word_df['decade'] == int(decade)].copy()
    try:
        decade_df['absolute_quartile'] = pd.qcut(
            decade_df['%_deviation_optimum'], q=4, labels=[0, 1, 2, 3])
    except (IndexError, ValueError):
        continue

    # Generate %Dev relative quartiles (quartiles across the word list in a decade)
    try:
        decade_df['relative_quartile'] = decade_df.groupby('Word List')['%_deviation_optimum'].transform(
            lambda x: pd.qcut(x, q=4, labels=[0, 1, 2, 3]))
    except (IndexError, ValueError):
        continue

    # Mark PNAS names
    decade_df['PNAS surname'] = decade_df['w'].isin(
        word_list_all['PNAS Asian Target Words'] + word_list_all['PNAS White Target Words'])

    surname_df = pd.concat([
        surname_df,
        decade_df.loc[decade_df['Word List'].isin(['White San Bruno All', 'Asian San Bruno All'])]])

g = sns.FacetGrid(
    surname_df, col="Word List", margin_titles=True, legend_out=True,
    hue='PNAS surname', palette='rocket_r')
g.set_axis_labels(x_var='Deviation from optimum (%) quartile')
g.set_titles(col_template="{col_name}")
g.map(sns.scatterplot, '%_deviation_optimum', 'decade')
for ax in g.axes.flatten():
    ax.set_xlabel("% Deviation from optimum\n(word-level)")
    ax.set_ylabel("Decade")
g.add_legend()
g.figure.savefig(os.path.join(output_path, 'Exp_surnames.png'))

# Regressions
regression_df = pd.read_csv(os.path.join(output_path, 'regression_df.csv'))

