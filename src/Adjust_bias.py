import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import src.bias_utils as bias_utils


def quartile_bias_scores(args, vectors, word_list_all, word_df):
    wls = ['Asian_San_Bruno_All', 'White_San_Bruno_All', 'PNAS Asian Target Words', 'PNAS White Target Words']
    main_df = pd.DataFrame()
    regression_df = pd.DataFrame()
    for decade, model in tqdm(vectors.items()):
        # Define otherization vectors
        otherization_idx = [model.key_to_index[w] for w in list(set(word_list_all['Otherization Words'])) if
                            bias_utils.present_word(model, w)]
        if args.match_garg:
            otherization_idx = [model.key_to_index[w] for w in word_list_all['Otherization Words'] if w in model]
        otherization_vecs = model.vectors[otherization_idx, :]

        # Absolute quartiles (quartiles across all word lists in a decade)
        decade_df = word_df.loc[word_df['decade'] == int(decade)].copy()
        try:
            decade_df['absolute_quartile'] = pd.qcut(
                decade_df[args.deviation], q=4, labels=[0, 1, 2, 3])
        except (IndexError, ValueError):
            continue

        # Generate %Dev relative quartiles (quartiles across the word list in a decade)
        try:
            decade_df['relative_quartile'] = decade_df.groupby('Word List')[args.deviation].transform(
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
                        attribute_vecs=otherization_vecs, t1_mean=white_mean_vec, t2_mean=asian_mean_vec,
                        cosine=args.cosine)

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
                                attribute_vecs=otherization_vecs, t1_mean=white_mean_vec, t2_mean=asian_mean_vec,
                                cosine=args.cosine)

                            # Append to main df
                            quart_dict = {
                                'Experiment': ['{}-{}'.format(quartile_type, quartile_type)], 'decade': [int(decade)],
                                'wl': ['{}'.format(wl)],
                                '%Deviation (quartile)': ['{}-{}'.format(quartile_i, quartile_j)],
                                'Bias score': [bias_score]
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
                        otherization_idx = [model.key_to_index[w] for w in list(set(otherization_dev_quart)) if
                                            bias_utils.present_word(model, w)]
                        if args.match_garg:
                            otherization_idx = [model.key_to_index[w] for w in otherization_dev_quart if
                                                w in model]
                        otherization_vecs = model.vectors[otherization_idx, :]

                        if asian_mean_vec is not None and white_mean_vec is not None and otherization_vecs is not None:
                            bias_score, _, _ = bias_utils.compute_bias_score(
                                attribute_vecs=otherization_vecs, t1_mean=white_mean_vec, t2_mean=asian_mean_vec,
                                cosine=args.cosine)

                            # Append to main df (Exp 4)
                            quart_dict = {
                                'Experiment': ['otherization'], 'decade': [int(decade)],
                                'wl': ['{}'.format(wl)],
                                '%Deviation (quartile)': ['{}-{}-{}'.format(quartile_a, quartile_w, quartile_o)],
                                'Bias score': [bias_score]
                            }
                            main_df = pd.concat([main_df, pd.DataFrame.from_dict(quart_dict)])

    return main_df


def regression_inputs(args, vectors, word_df):
    # Generate bias scores by quartile
    regression_df = pd.DataFrame()
    for decade, model in tqdm(vectors.items()):
        decade_df = word_df.loc[word_df['decade'] == int(decade)].copy()

        # Exp 5: Regression inputs
        for wl in ['{}_San_Bruno_All', 'PNAS {} Target Words']:

            asian_df = decade_df.loc[decade_df['Word List'] == wl.format('Asian').replace('_', ' ')].copy()
            white_df = decade_df.loc[decade_df['Word List'] == wl.format('White').replace('_', ' ')].copy()
            group_dict = {'Asian': asian_df, 'White': white_df}
            otherization_df = decade_df.loc[decade_df['Word List'] == 'Otherization Words'].copy()

            # Single surnames
            for other_w in otherization_df['w'].unique():
                # Define otherization words
                otherization_idx = [model.key_to_index[w] for w in [other_w] if bias_utils.present_word(model, w)]
                if args.match_garg:
                    otherization_idx = [model.key_to_index[w] for w in [other_w] if w in model]
                otherization_vecs = model.vectors[otherization_idx, :]

                if otherization_vecs.shape[0] == 0:
                    continue

                # Need to normalize in case we're using SPPMI vectors or SGNS vectors
                otherization_vecs = otherization_vecs / np.linalg.norm(otherization_vecs, axis=1)

                # Otherization word dict
                other_w_dict = otherization_df.loc[otherization_df['w'] == other_w]

                for group, group_df in group_dict.items():
                    surnames = group_df['w'].unique()
                    for surname in surnames:
                        # Surnames (Note there is no need to add pre-normalization here because
                        # we're using a single name)
                        surname_vec = bias_utils.compute_mean_vector(model, [surname], pre_normalize=True)

                        if surname_vec is None:
                            continue

                        # Compute bias components
                        _, t1_component, _ = bias_utils.compute_bias_score(
                            attribute_vecs=otherization_vecs, t1_mean=surname_vec, t2_mean=surname_vec,
                            cosine=args.cosine)

                        # Surname dict
                        surname_dict = group_df.loc[group_df['w'] == surname]

                        quart_dict = {
                            'Experiment': ['regression'], 'decade': [int(decade)], 'wl': ['{}'.format(wl)],
                            'other_word': [other_w],
                            'surname': [surname],
                            'Metric': [t1_component.item()], 'Group': [group],
                            '%Deviation Target (quartile)': [None],
                            '%Deviation Target': [surname_dict[args.deviation].iloc[0]],
                            '%Deviation Otherization': [other_w_dict[args.deviation].iloc[0]],
                            '%Deviation Otherization (quartile)': [None],
                            'type': ['single_surname'], 'bias_score': [None]
                        }
                        regression_df = pd.concat([regression_df, pd.DataFrame.from_dict(quart_dict)])

            # Aggregate surnames
            # Define otherization words
            otherization_idx = [model.key_to_index[w] for w in otherization_df['w'].unique() if bias_utils.present_word(model, w)]
            if args.match_garg:
                otherization_idx = [model.key_to_index[w] for w in otherization_df['w'].unique() if w in model]
            otherization_vecs = model.vectors[otherization_idx, :]

            # Need to normalize in case we're using SPPMI vectors
            if otherization_vecs.shape[0] != 0:
                otherization_vecs = otherization_vecs / np.linalg.norm(otherization_vecs, axis=1).reshape(-1, 1)

            asian_surnames = list(asian_df['w'].unique())
            white_surnames = list(white_df['w'].unique())

            asian_mean_vec = bias_utils.compute_mean_vector(model, asian_surnames, pre_normalize=True)
            white_mean_vec = bias_utils.compute_mean_vector(model, white_surnames, pre_normalize=True)

            if asian_mean_vec is not None and white_mean_vec is not None and otherization_vecs.shape[0] != 0:
                bias_score, _, _ = bias_utils.compute_bias_score(
                    attribute_vecs=otherization_vecs, t1_mean=white_mean_vec, t2_mean=asian_mean_vec, cosine=args.cosine)

                # Append to main df (Exp 5)
                quart_dict = {
                    'Experiment': ['regression'], 'decade': [int(decade)], 'wl': ['{}'.format(wl)],
                    'other_word': ['--Overall--'], 'surname': [None],
                    'Metric': [None], 'Group': [None],
                    '%Deviation Target (quartile)': [None],
                    '%Deviation Target': [None],
                    '%Deviation Otherization': [None],
                    '%Deviation Otherization (quartile)': [None],
                    'type': ['aggregated'], 'bias_score': [bias_score]
                }
                regression_df = pd.concat([regression_df, pd.DataFrame.from_dict(quart_dict)])

    # Add norms
    if args.vectors == 'SGNS':
        norm_df = pd.read_csv(args.norm_dir)
        norm_df = norm_df.loc[(norm_df['k'] == args.negative) & (norm_df['d'] == args.d)]

        # Add norm for otherization word
        regression_df = regression_df.merge(
            norm_df[['word', 'norm', 'decade']],
            left_on=['decade', 'other_word'], right_on=['decade', 'word'], validate='many_to_one', how='left')
        regression_df.rename(columns={'norm': 'norm_other'}, inplace=True)
        regression_df.drop(['word'], axis=1, inplace=True)

        # Add norm for surname
        regression_df = regression_df.merge(
            norm_df[['word', 'norm', 'decade']],
            left_on=['decade', 'surname'], right_on=['decade', 'word'], validate='many_to_one', how='left')
        regression_df.rename(columns={'norm': 'norm_surname'}, inplace=True)
        regression_df.drop(['word'], axis=1, inplace=True)

        regression_df.loc[regression_df['type'] == 'aggregated', 'norm_other'] = None
        regression_df.loc[regression_df['type'] == 'aggregated', 'norm_surname'] = None

    return regression_df


def main(args):
    # Load word_df and vectors
    word_df = pd.read_csv(os.path.join(args.input_path, 'word_df_median.csv'))
    word_df = word_df.loc[(word_df['k'] == args.negative) & (word_df['d'] == args.d)]

    if args.vectors == 'HistWords':
        vectors = bias_utils.load_coha(input_dir=args.vector_path)
    elif args.vectors == 'SPPMI':
        vectors = bias_utils.load_SPPMI(input_dir=args.vector_path, negative=args.negative)
    elif args.vectors == 'SGNS':
        vectors = bias_utils.load_coha_SGNS(
            input_dir=args.vector_path, negative=args.negative, d=args.d, norm=False)
    else:
        raise Exception('Check vectors')

    # Load word lists
    with open('../../Local/word_lists/word_lists_all.json', 'r') as file:
        word_list_all = json.load(file)

    # Generate bias scores by quartile
    if False:
        main_df = quartile_bias_scores(args=args, vectors=vectors, word_list_all=word_list_all, word_df=word_df)
        main_df['wl'] = main_df['wl'].apply(lambda wl: wl.replace('_', ' '))
        main_df.to_csv(os.path.join(
            args.output_path, 'quartile_df_MG{}_CS{}.csv'.format(args.match_garg, args.cosine)), index=False)

    # Regression inputs
    print('[INFO] Generating inputs for regression')
    regression_df = regression_inputs(args=args, vectors=vectors, word_df=word_df)
    regression_df['wl'] = regression_df['wl'].apply(lambda wl: wl.replace('_', ' '))
    regression_df.to_csv(os.path.join(
        args.output_path, f"regression_df_MG{args.match_garg}_CS{args.cosine}_k{args.negative}_d{args.d}.csv"),
        index=False)

    # Plot
    if args.visualize:
        main_df = pd.read_csv(os.path.join(args.output_path, 'quartile_df_MG{}_CS{}.csv'.format(args.match_garg, args.cosine)))

        # Plot Ex 1
        exp1 = main_df.loc[(main_df['Experiment'] == 'relative-fixed') & (main_df['decade'] > 1890)].copy()
        g = sns.FacetGrid(
            exp1, col="wl", col_wrap=2, margin_titles=True, legend_out=True,
            hue='decade', palette='rocket_r')
        g.set_axis_labels(x_var='Deviation from optimum (%) quartile')
        g.set_titles(col_template="{col_name}")
        g.map(sns.lineplot, '%Deviation (quartile)', 'Bias score')
        g.add_legend()
        g.figure.savefig(os.path.join(args.output_path, 'Exp1_bias_corr_quartiles.png'))

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
                g.figure.savefig(os.path.join(args.output_path, 'Exp_{}_{}_bias_corr_quartiles.png'.format(quartile_type, wl)))

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
            g.figure.savefig(os.path.join(args.output_path, 'Exp4_{}.png'.format(wl.replace('{}', '').replace(' ', ''))))

        # Explore PNAS vs SBruno
        surname_df = pd.DataFrame()
        for decade, model in vectors.items():

            # Absolute quartiles (quartiles across all word lists in a decade)
            decade_df = word_df.loc[word_df['decade'] == int(decade)].copy()
            try:
                decade_df['absolute_quartile'] = pd.qcut(
                    decade_df[args.deviation], q=4, labels=[0, 1, 2, 3])
            except (IndexError, ValueError):
                continue

            # Generate %Dev relative quartiles (quartiles across the word list in a decade)
            try:
                decade_df['relative_quartile'] = decade_df.groupby('Word List')[args.deviation].transform(
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
        g.map(sns.scatterplot, args.deviation, 'decade')
        for ax in g.axes.flatten():
            ax.set_xlabel("% Deviation from optimum\n(word-level)")
            ax.set_ylabel("Decade")
        g.add_legend()
        g.figure.savefig(os.path.join(args.output_path, 'Exp_surnames.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # The Garg et al. computation does not consider whether otherization vectors are non-zero in a model.
    parser.add_argument("-match_garg", type=bool, default=False)
    parser.add_argument("-cosine", type=bool, required=True)
    parser.add_argument("-visualize", type=bool, default=False)
    parser.add_argument("-input_path", type=str, required=False)
    parser.add_argument("-output_path", type=str, required=False)
    parser.add_argument("-vector_path", type=str, required=False)
    parser.add_argument("-negative", type=int, required=False)
    parser.add_argument("-d", type=int, required=False)
    parser.add_argument("-vectors", type=str, required=True)
    parser.add_argument("-deviation", type=str)
    parser.add_argument("-norm_dir", type=str, required=False)

    args = parser.parse_args()

    # Paths
    args.input_path = f'../results/{args.vectors}/PMI'
    if args.vectors == 'SPPMI':
        # Note: for SPPMI, we use %D-SPPMI-PMI from the word_median_df, which only depends on k.
        args.input_path = f'../results/SGNS/PMI'
    args.output_path = f'../results/{args.vectors}/regressions'
    os.makedirs(args.output_path, exist_ok=True)

    if args.vectors == 'HistWords':
        args.vector_path = '../../Replication-Garg-2018/data/coha-word'
    elif args.vectors == 'SPPMI':
        args.vector_path = '../results/SPPMI/vectors'
    elif args.vectors == 'SGNS':
        args.vector_path = '../../COHA-SGNS/results/vectors'
        args.norm_dir = '../../COHA-SGNS/results/norm/SGNS-norms.csv'
    else:
        raise Exception('Check vectors')

    # Deviation
    args.deviation = '%D-SPPMI-PMI' if args.vectors == 'SPPMI' else '%D-dot-PMI'
    print(f'[INFO] Using deviation: {args.deviation}')

    args.negative = 5
    args.d = 300
    print(f'[INFO] Using negative sampling parameter: {args.negative} and dimension: {args.d}')

    main(args)
