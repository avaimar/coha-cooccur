import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import pandas as pd
import seaborn as sns
from tqdm import tqdm

tqdm.pandas()


# Functions
def sigmoid(x):
    return 1/(1 + np.exp(-x))


def equation5(hash_wc, dot_wc, k, hash_w, hash_c, D_size):
    if D_size == 0:
        return None
    eq5 = hash_wc * np.log(sigmoid(dot_wc))
    eq5 += k * hash_w * hash_c * np.log(sigmoid(-dot_wc)) / D_size
    return eq5


def main(args):
    # Load PMI data
    if args.vectors == 'HistWords':
        pmi_df = pd.read_csv(os.path.join(args.output_dir, 'PMI', 'pmi.csv'))
        pmi_df['k'] = 5
        pmi_df['d'] = 300
    elif args.vectors == 'SGNS':
        pmi_files = glob.glob(os.path.join(args.output_dir, '**', 'pmi.csv'), recursive=True)
        pmi_df = pd.DataFrame()
        for file in pmi_files:
            name = file.split(os.path.sep)[-3].replace('SGNS-', '')
            k, d = name.split('-')
            if k == '0' or d != '300':  # PMI only varies with k, not d
                continue
            comb_df = pd.read_csv(file)
            comb_df['k'] = int(k)
            comb_df['d'] = int(d)
            pmi_df = pd.concat([pmi_df, comb_df])
    else:
        raise Exception('Check embeddings.')

    # Standardize Word List names
    pmi_df.rename(columns={'word_list': 'Word List'}, inplace=True)
    pmi_df['Word List'] = pmi_df['Word List'].apply(lambda wl: wl.replace('_', ' '))

    # Compute SPPMI (eq. 12) (note that PMI column is already shifted by log k)
    pmi_df['SPPMI'] = pmi_df['PMI'].apply(lambda spmi: max(spmi, 0))

    # Compute eq 5.
    pmi_df['eq5_dot'] = pmi_df.progress_apply(
        lambda row: equation5(
            hash_wc=row['#wc'], dot_wc=row['w_dot_c'], k=row['k'],
            hash_w=row['#w'], hash_c=row['#c'], D_size=row['D']), axis=1)

    pmi_df['eq5_SPPMI'] = pmi_df.progress_apply(
        lambda row: equation5(
            hash_wc=row['#wc'], dot_wc=row['SPPMI'], k=row['k'],
            hash_w=row['#w'], hash_c=row['#c'], D_size=row['D']), axis=1)

    pmi_df['eq5_PMI'] = pmi_df.progress_apply(
        lambda row: equation5(
            hash_wc=row['#wc'], dot_wc=row['PMI'], k=row['k'],
            hash_w=row['#w'], hash_c=row['#c'], D_size=row['D']), axis=1)

    # Get % optimal
    pmi_df['%D-dot-PMI'] = pmi_df.progress_apply(
        lambda row: (row['eq5_dot'] - row['eq5_PMI']) / row['eq5_PMI'] if
        row['eq5_PMI'] != 0 else None, axis=1)

    pmi_df['%D-SPPMI-PMI'] = pmi_df.progress_apply(
        lambda row: (row['eq5_SPPMI'] - row['eq5_PMI']) / row['eq5_PMI'] if
        row['eq5_PMI'] != 0 else None, axis=1)

    pmi_df['%D-dot-SPPMI'] = pmi_df.progress_apply(
        lambda row: (row['eq5_dot'] - row['eq5_SPPMI']) / row['eq5_SPPMI'] if
        row['eq5_SPPMI'] != 0 else None, axis=1)

    # Drop duplicates and save
    pmi_df.drop_duplicates(
        subset=['w_idx', 'c_idx', 'Word List', 'decade', 'k', 'd'], inplace=True)
    if not os.path.exists(os.path.join(args.output_dir, 'PMI')):
        os.makedirs(os.path.join(args.output_dir, 'PMI'))
    pmi_df.to_csv(os.path.join(args.output_dir, 'PMI', 'pmi_eq5.csv'), index=False)

    # Compute word-level measure of deviation so we can relate this to the bias scores
    word_df = pmi_df.copy()
    word_df = word_df.groupby(['w_idx', 'w', 'decade', 'Word List', 'k', 'd']).agg(
        {'%D-dot-PMI': args.word_stat, '%D-SPPMI-PMI': args.word_stat}).reset_index()

    word_df.to_csv(os.path.join(
        args.output_dir, 'PMI', 'word_df_{}.csv'.format(args.word_stat)), index=False)

    if args.plot:
        # *** Plot % deviation from optimum
        sns.set_theme(style="white", font_scale=1.8)
        g = sns.FacetGrid(
            pmi_df, row="Word List", margin_titles=True, legend_out=True,
            hue='Word List', height=5.5, aspect=3.5,
            row_order=['Asian San Bruno All', 'PNAS Asian Target Words',
                       'White San Bruno All',  'PNAS White Target Words', 'Otherization Words'])
        g.map(sns.kdeplot, '%_deviation_optimum', linewidth=4)
        g.set_axis_labels(x_var='Deviation from optimum (%)')
        g.set_titles(row_template="{row_name}")
        g.figure.savefig(os.path.join(args.output_dir, 'density_%dev.png'))

        # *** Plot % deviation from optimum (at the word level)
        g = sns.FacetGrid(
            word_df, row="Word List", margin_titles=True, legend_out=True,
            hue='Word List', height=5.5, aspect=3.5,
            row_order=['Asian San Bruno All', 'PNAS Asian Target Words',
                       'White San Bruno All',  'PNAS White Target Words', 'Otherization Words'])
        g.set_axis_labels(x_var='Deviation from optimum (%)')
        g.map(sns.kdeplot, '%_deviation_optimum', linewidth=4)
        g.figure.savefig(os.path.join(args.output_dir, 'density_%dev_word_{}.png'.format(args.word_stat)))

    if args.plot:
        # SPPMI
        slist = {'PNAS White Target Words': 'PNAS', 'PNAS Asian Target Words': 'PNAS',
                 'White San Bruno All': 'San Bruno', 'Asian San Bruno All': 'San Bruno',
                 'Otherization Words': None}
        stype = {'PNAS White Target Words': 'White', 'PNAS Asian Target Words': 'Asian',
                 'White San Bruno All': 'White', 'Asian San Bruno All': 'Asian',
                 'Otherization Words': None}
        word_df['Surname List'] = word_df['Word List'].apply(lambda wl: slist[wl])
        word_df['Surname Type'] = word_df['Word List'].apply(lambda wl: stype[wl])

        sppmi_df = word_df.loc[
            (word_df['Word List'] != 'Otherization') & (word_df['decade'] == 1990)].copy()

        g = sns.FacetGrid(
            sppmi_df.loc[(sppmi_df['k'] == 5) & (sppmi_df['d'] == 100)],
            #sppmi_df,
            margin_titles=True, legend_out=True,
            hue='Surname Type', col='Surname List')
        g.map(sns.kdeplot, '%D-SPPMI-PMI', linewidth=1)
        g.set_axis_labels(x_var='Deviation (%)')
        g.set_titles(col_template="{col_name}")
        plt.legend(title='Word List')
        g.figure.savefig(os.path.join(args.output_dir, 'PMI', 'density_%dev_word_median-SPPMI.png'))

        g = sns.FacetGrid(
            sppmi_df.loc[(sppmi_df['k'] == 5) & (sppmi_df['d'] == 100)],
            #sppmi_df,
            margin_titles=True, legend_out=True,
            hue='Surname Type', col='Surname List')
        g.map(sns.kdeplot, '%D-dot-PMI', linewidth=1)
        g.set_axis_labels(x_var='Deviation (%)')
        g.set_titles(col_template="{col_name}")
        plt.legend(title='Word List')
        g.figure.savefig(os.path.join(args.output_dir, 'PMI', 'density_%dev_word_median-TEST.png'))


        melt_df = sppmi_df.melt(
            id_vars=['w_idx', 'w', 'decade', 'k', 'd', 'Word List', 'Surname List', 'Surname Type'],
            value_vars=['%_deviation_optimum', '%_deviation_optimum-SPPMI-PMI'],
            var_name=['Deviation type'], value_name='Deviation (%)')
        melt_df['Deviation type'] = melt_df['Deviation type'].apply(
            lambda d: 'Dot-PMI' if d == '%D-dot-PMI' else 'SPPMI-PMI')

        # Keep only one dimension for SPPMI
        melt_df = melt_df.loc[(melt_df['k'] == 5) & (melt_df['d'] == 500)]

        g = sns.FacetGrid(
            melt_df,
            row='Deviation type', sharey=False, #sharex=False,
            margin_titles=True, legend_out=True,
            hue='Surname Type', col='Surname List')
        g.map(sns.kdeplot, 'Deviation (%)', linewidth=1)
        g.set_axis_labels(x_var='Deviation (%)')
        g.set_titles(col_template="{col_name}", row_template='{row_name}')
        g.add_legend(title='Word List')
        g.figure.savefig(os.path.join(output_dir, 'PMI', 'density_%dev_word_median-SPPMI.png'))










    # Average Asian / White deviation across k, d
    wl_df = word_df.groupby(['k', 'd', 'Word List', 'decade']).agg(
        {'%_deviation_optimum': args.word_stat,
         '%_deviation_optimum-SPPMI': args.word_stat}).reset_index()
    for dev in ['', '-SPPMI']:
        for wl in wl_df['Word List'].unique():
            plt.clf()
            ax = sns.scatterplot(wl_df.loc[wl_df['Word List'] == wl],
                            x='k', y=f'%_deviation_optimum{dev}', hue='d')
            ax.set(
                ylabel='Median word-level deviation from optimum (%)', xlabel='Number of negative samples')
            plt.legend(title='Dimensionality')
            ax.set_xticks(range(5, 30, 5))
            ax.set_title(f'{wl}')
            ax.figure.savefig(os.path.join(
                args.output_dir, 'PMI', f'kd_plot-{wl}-{args.word_stat}{dev}.png'))

    wl_df = wl_df.melt(
        id_vars=['k', 'd', 'Word List', 'decade'],
        value_vars=['%_deviation_optimum', '%_deviation_optimum-SPPMI'],
        var_name='Type', value_name='% Deviation')
    wl_df['Type'] = wl_df['Type'].apply(lambda t: 'Deviation % (PMI)' if t == '%_deviation_optimum' else 'Deviation % (SPPMI)')

    for wl in wl_df['Word List'].unique():
        sns.color_palette("rocket")
        g = sns.FacetGrid(
            wl_df.loc[wl_df['Word List'] == wl], col="Type", margin_titles=True, legend_out=True,
            hue='d', height=3.7, aspect=1.2, palette="rocket_r")
        g.map(sns.scatterplot, 'k', '% Deviation', palette="rocket_r")
        g.set(xticks=range(5, 30, 5))
        g.set_axis_labels(y_var='Median word-level deviation from optimum (%)', x_var='Number of negative samples')
        g.set_titles(col_template="{col_name}")
        g.fig.suptitle(f'{wl}')
        plt.legend(title='Dimensionality')
        g.figure.savefig(os.path.join(args.output_dir, 'PMI', f'kd_plot_grid-{wl}-{args.word_stat}.png'))

    for wl in wl_df['Word List'].unique():
        sns.color_palette("rocket")
        g = sns.FacetGrid(
            wl_df.loc[wl_df['Word List'] == wl], col="Type", margin_titles=True, legend_out=True,
            hue='k', height=3.7, aspect=1.2, palette="rocket_r")
        g.map(sns.scatterplot, 'd', '% Deviation', palette="rocket_r")
        g.set(xticks=[100, 300, 500, 1000])
        g.set_axis_labels(y_var='Median word-level deviation from optimum (%)', x_var='Number of negative samples')
        g.set_titles(col_template="{col_name}")
        g.fig.suptitle(f'{wl}')
        plt.legend(title='Dimensionality')
        g.figure.savefig(os.path.join(args.output_dir, 'PMI', f'kd_plot_grid-{wl}-{args.word_stat}.png'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-vectors", type=str)
    parser.add_argument("-output_dir", type=str)
    parser.add_argument("-word_stat", type=str)
    parser.add_argument("-plot", type=bool, default=False)

    args = parser.parse_args()

    # Paths
    args.output_dir = f'../results/{args.vectors}/'

    # Params
    args.word_stat = 'median'
