"""Define the model class for the variant identification objective."""

# change the color profile

import numpy as np
from badgers_mod.utils import prepare_sequences as prep_seqs
from badgers_mod.utils import cas13_cnn as cas13_cnn
import tensorflow as tf

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import time




grid = {'c': 1.0, 'a': 3.769183, 'k': -3.833902, 'o': -2.134395, 't2w': 2.973052}

# heatmap parameter
cell_size = 0.4 # in the heatmap (inch)
color_map_type = "YlGnBu"#"viridis" # or "YlGnBu"

# gen_guide, target_set1, target_set2
gen_guide_file = '/Users/soonwoohong/Library/CloudStorage/GoogleDrive-sh5230@princeton.edu/.shortcut-targets-by-id/1fc61rmoxxDIgSxNlbKpHNutryvyYltP4/Soonwoo/python/badgers_search_guide/crRNA_search_results_1.xlsx'
target_list_file = '/Users/soonwoohong/Library/CloudStorage/GoogleDrive-sh5230@princeton.edu/.shortcut-targets-by-id/1fc61rmoxxDIgSxNlbKpHNutryvyYltP4/Soonwoo/Projects/TB DST - CRISPR/sequence data/INH/inhA/results/one-vs-others/INH_inhA_targets_incl_WT.xlsx'
gen_guide_data = pd.read_excel(gen_guide_file, header=0)
# this target file should have WT sequence if you want to include it
target_data = pd.read_excel(target_list_file, header=0)


score_data = []

for file_idx in range(len(gen_guide_data)):
    #start = time.perf_counter()

    start_pos = gen_guide_data.start_pos[file_idx]
    guide_of_interest = [gen_guide_data.guide_sequence[file_idx]]
    start_pos = gen_guide_data.start_pos[file_idx]
    seqs = [seq.upper() for seq in target_data.new_seq]
    # The predictive models require 10 nt of context on each side of the 28 nt probe-binding region
    context_nt = 10
    seqs_frag = [s[start_pos - context_nt-1:start_pos - context_nt-1 + 48] for s in seqs]
    seqs_onehot = [prep_seqs.one_hot_encode(x) for x in seqs_frag]
    gen_guide_onehot = [prep_seqs.one_hot_encode(x) for x in guide_of_interest]

    pred_perf_seqs, pred_act_seqs = cas13_cnn.run_full_model(gen_guide_onehot, seqs_onehot)
    pos4 = tf.constant(4.0)
    weighted_perf_seqs = tf.math.subtract(tf.math.multiply(pred_act_seqs, tf.math.add(pred_perf_seqs, pos4)), pos4)

    score_data.append(weighted_perf_seqs.numpy().tolist()[0])
    #end = time.perf_counter()
    #elapsed_seconds = end - start
    #print(f"Elapsed: {elapsed_seconds:.6f} s")


score_pd = pd.DataFrame(score_data, columns = target_data.new_name.to_list())
exp_data = pd.concat([gen_guide_data,score_pd],axis=1)

exp_data.to_excel(gen_guide_file[0:-5]+"_score.xlsx")

#making heatmap
final_data = np.array(score_data)
final_data_subtracted = final_data - np.tile(gen_guide_data.mean_on_target_act.to_numpy()[:,np.newaxis], (1, len(target_data)))
fig, ax = plt.subplots(figsize=(cell_size*gen_guide_data.shape[0]+3,
                                cell_size*gen_guide_data.shape[1]+1),
                                dpi=300)

sns.heatmap(final_data.transpose(), cmap=color_map_type,
            square = True,
            ax = ax, linewidth = 0.3, linecolor = 'black')
plt.show()
fig.savefig(gen_guide_file[0:-5]+"_score.png", bbox_inches='tight', dpi=300)

for gap in ([0.1, 0.2, 0.3, 0.4, 0.5]):
    fig, ax = plt.subplots(figsize=(cell_size * gen_guide_data.shape[0] + 3,
                                    cell_size * gen_guide_data.shape[1] + 1),
                           dpi=300)

    bounds = [-gap*4, -gap*3, -gap*2, -gap*1, 0.0001, 100]
    colors = [(8,81,156),(49,130,189), (107,174,214), (189,215,231), (178,24,43)]  # One color per bin
    normalized_colors = [(r/255, g/255, b/255) for r, g, b in colors]

    cmap = ListedColormap(normalized_colors)
    norm = BoundaryNorm(bounds, len(colors))

    sns.heatmap(final_data_subtracted.transpose(), cmap=cmap, norm=norm, ax = ax, linewidth = 0.3, linecolor = 'black',
                square = True)
    plt.show()
    fig.savefig(gen_guide_file[0:-5]+"_gap_"+str(gap).replace(".",",")+".png", bbox_inches='tight', dpi=300)

