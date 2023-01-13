import matplotlib.pyplot as plt
import numpy as np
import os
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


# Params
K = 5
word_stat = 'max'

# Paths
output_path = '../results/PMI'

# Load PMI data
pmi_df = pd.read_csv(os.path.join(output_path, 'pmi.csv'))

# Standardize Word List names
pmi_df.rename(columns={'word_list': 'Word List'}, inplace=True)
pmi_df['Word List'] = pmi_df['Word List'].apply(lambda wl: wl.replace('_', ' '))

# Compute eq 5.
pmi_df['eq5_dot'] = pmi_df.progress_apply(
    lambda row: equation5(
        hash_wc=row['#wc'], dot_wc=row['w_dot_c'], k=K,
        hash_w=row['#w'], hash_c=row['#c'], D_size=row['D']),
    axis=1)

pmi_df['eq5_pmi-opt'] = pmi_df.progress_apply(
    lambda row: equation5(
        hash_wc=row['#wc'], dot_wc=row['PMI'], k=K,
        hash_w=row['#w'], hash_c=row['#c'], D_size=row['D']),
    axis=1)

# Get % optimal
pmi_df['%_deviation_optimum'] = pmi_df.progress_apply(
    lambda row: (row['eq5_dot'] - row['eq5_pmi-opt']) / row['eq5_pmi-opt'] if
    row['eq5_pmi-opt'] != 0 else None,
    axis=1)

# Compute word-level measure of deviation so we can relate this to the bias scores
word_df = pmi_df.copy()
pmi_df.to_csv(os.path.join(output_path, 'pmi_eq5.csv'), index=False)
word_df = word_df.groupby(['w_idx', 'w', 'decade', 'Word List']).agg(
    {'%_deviation_optimum': word_stat}).reset_index()
word_df.to_csv(os.path.join(output_path, 'word_df_{}.csv'.format(word_stat)), index=False)

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
g.figure.savefig(os.path.join(output_path, 'density_%dev.png'))

# *** Plot % deviation from optimum (at the word level)
g = sns.FacetGrid(
    word_df, row="Word List", margin_titles=True, legend_out=True,
    hue='Word List', height=5.5, aspect=3.5,
    row_order=['Asian San Bruno All', 'PNAS Asian Target Words',
               'White San Bruno All',  'PNAS White Target Words', 'Otherization Words'])
g.set_axis_labels(x_var='Deviation from optimum (%)')
g.map(sns.kdeplot, '%_deviation_optimum', linewidth=4)
g.figure.savefig(os.path.join(output_path, 'density_%dev_word_{}.png'.format(word_stat)))







# Plot eq. 5
#X_plot = np.linspace(pmi_df['eq5_dot'].min(), pmi_df['eq5_dot'].max(), 100)
#Y_plot = X_plot

g = sns.FacetGrid(pmi_df, col="decade", col_wrap=5, margin_titles=True, legend_out=True)
g.map(sns.scatterplot, "eq5_dot", "eq5_pmi-opt", 'Word List', ).set(
    yscale='symlog', xscale='symlog')
g.add_legend()
#axes = g.axes
#for ax in axes:
#    ax.plot(X_plot, X_plot)
#g.set(xlabel='')
g.figure.savefig(os.path.join(output_path, 'decadal_loss.png'))
plt.clf()



# in a decade
ax = sns.scatterplot(
    pmi_df.loc[pmi_df['decade'] == 1880],
    x="eq5_dot", y="eq5_pmi-opt", hue='Word List')
#ax.plot(X_plot, X_plot)

sns.kdeplot(pmi_df.loc[pmi_df['decade'] == 1880], x='eq5_dot', hue='Word List')
plt.xscale('symlog')


# Plot dot vs pmi
g = sns.FacetGrid(pmi_df, col="decade", col_wrap=5, margin_titles=True, legend_out=True)
#X_plot = np.linspace(pmi_df['PMI'].min(), pmi_df['PMI'].max(), 100)
#Y_plot = X_plot
g.add_legend()
#axes = g.axes
#for ax in axes:
    #ax.plot(X_plot, X_plot)

g.map(sns.scatterplot, "w_dot_c", "PMI", 'Word List', )#.set(
    #yscale='symlog', xscale='symlog')

#g.set(xlabel='')
g.figure.savefig(os.path.join(output_path, 'decadal_pmi.png'))
plt.clf()

