"""
This is a code for searching guide sequence based on the ADAPT model.
Input:
target sequence
(optional) guide sequence (starting)


"""

# change the color profile

import numpy as np
from badgers_mod.utils import prepare_sequences as prep_seqs
import tensorflow as tf
from badgers_mod.utils import cas13_cnn as cas13_cnn
import pandas as pd


import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import time

from Bio import AlignIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from io import StringIO
from Bio import Align

# adapt prediction parameters (don't change unless necessary)
grid = {'c': 1.0, 'a': 3.769183, 'k': -3.833902, 'o': -2.134395, 't2w': 2.973052}
context_nt = 10
pos4 = tf.constant(4.0)

# search parameter
scan_size = 10
top_k_crRNA = 10


# heatmap parameter
cell_size = 0.4 # in the heatmap (inch)
color_map_type = "YlGnBu"#"viridis" # or "YlGnBu"

# gen_guide, target_set1, target_set2
gen_guide_file = './cluster1-3_final_merge.xlsx'
target_list_file = './INH_inhA_targets_incl_WT.xlsx'
gen_guide_data = pd.read_excel(gen_guide_file, header=0)

target_data = pd.read_excel(target_list_file, header=0)
target_seqs = target_data.new_seq.to_list()
WT_seq = target_seqs[0] # change this in case that you have WT (or base seq) not in the first row.
target_name = target_data.new_name.to_list()

mm_coordinates = []
aligner = Align.PairwiseAligner()

# check where mismatch starts and ends
for i in range(len(target_seqs)-1):
    ii = i+1
    alignments = aligner.align(WT_seq, target_seqs[ii])
    alignment = alignments[0]
    mm_start = alignment.aligned[0][0][1]
    mm_end = alignment.aligned[0][-1][0]

    # mm_coordinates = [(index, mismatch_start, mismatch_end)]
    mm_coordinates.append((ii, mm_start, mm_end))


# group them
cluster_size = 20
mm_coordinates.sort(key=lambda x: x[1]) # sort by start coord
clusters = []
current_cluster = []
anchor_start = None # first start in the current cluster

for idx, start, end in mm_coordinates:
    if anchor_start is None:
        # starting first cluster
        current_cluster = [idx]
        anchor_start = start
    elif end - anchor_start <= cluster_size:
        # close enough to current cluster
        current_cluster.append(idx)
    else:
        # too far → commit old cluster, start new one
        clusters.append(current_cluster)
        current_cluster = [idx]
        anchor_start = start

# add the last cluster
if current_cluster:
    clusters.append(current_cluster)

# search crRNAs
selected_crRNAs = []

for cluster in clusters:
    cluster_start = mm_coordinates[cluster[0]-1][1]
    cluster_end = mm_coordinates[cluster[-1]-1][2]
    print(f"Cluster {cluster} starts at {cluster_start} and ends at {cluster_end}")
    print(f"Target sequences: {[target_name[i] for i in cluster]}")
    print("--------------------------------")
    search_start = cluster_start - scan_size
    search_end = cluster_end + scan_size
    print(f"Search region starts at {search_start} and ends at {search_end}")
    cluster = [0] + cluster # put the wt sequence to the head of list
    seqs = [target_seqs[cluster_idx].upper() for cluster_idx in cluster]

    for start_pos in range(search_start-28, search_end-28): # -28 because of crRNA spacer length
        for interest in range(len(seqs)):
            seq_of_interest = seqs[interest]
            idx_of_interest = cluster[interest]
        start_guide = [seq_of_interest[start_pos-1:start_pos-1+28]]

        # The predictive models require 10 nt of context on each side of the 28 nt probe-binding region
        seqs_frag = [s[start_pos - context_nt-1:start_pos - context_nt-1 + 48] for s in seqs]
        seqs_onehot = [prep_seqs.one_hot_encode(x) for x in seqs_frag]
        gen_guide_onehot = [prep_seqs.one_hot_encode(x) for x in start_guide]
        pred_perf_seqs, pred_act_seqs = cas13_cnn.run_full_model(gen_guide_onehot, seqs_onehot)
        weighted_perf_seqs = tf.math.subtract(tf.math.multiply(pred_act_seqs, tf.math.add(pred_perf_seqs, pos4)), pos4)
        weighted_perf_seqs_np = weighted_perf_seqs[0].numpy()
        # score
        # instead of just subtracting the mean off-target from the on-target,
        # put the additional penalty of top_off_target.
        # later, I can add some weights to increase either of penalty.
        on_target_act = weighted_perf_seqs_np[interest]
        top_off_target_act = np.max(weighted_perf_seqs_np[weighted_perf_seqs_np != on_target_act])
        mean_off_target_act = np.mean(weighted_perf_seqs_np[weighted_perf_seqs_np != on_target_act])
        temp_score = on_target_act - top_off_target_act-mean_off_target_act


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

def align_seq():

    return