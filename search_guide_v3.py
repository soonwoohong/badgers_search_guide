"""
+batch mode
+GPU or multiple CPU options

This is a code for searching guide sequence based on the ADAPT model.

calculate a batch of guides


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

import itertools
from Bio import Align

# adapt prediction parameters (don't change unless necessary)
grid = {'c': 1.0, 'a': 3.769183, 'k': -3.833902, 'o': -2.134395, 't2w': 2.973052}
context_nt = 10
pos4 = tf.constant(4.0)

# search parameter
scan_size = 10
top_k_crRNA = 4
max_mismatches = 3
cut_off = -2
BATCH_SIZE = 512

# heatmap parameter
cell_size = 0.4 # in the heatmap (inch)
color_map_type = "YlGnBu"#"viridis" # or "YlGnBu"

# gen_guide, target_set1, target_set2
#gen_guide_file = './cluster1-3_final_merge.xlsx'
target_list_file = './INH_inhA_targets_incl_WT.xlsx'
#gen_guide_data = pd.read_excel(gen_guide_file, header=0)

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

DNA_bases = "ACGT"

# make intentional mismatch
def generate_intentional_mismatches(guide: str,
                                    num_mismatches: int,
                                    alphabet=DNA_bases):
    """
    Yield the original guide first, then mutated guides with up to `max_mismatches`.
    """
    L = len(guide)
    positions = range(L)

    # always yield original
    yield guide

    # all unordered pairs
    k = num_mismatches
    #for k in range(1, max_mismatches + 1):
        # choose which positions to mutate
    for pos_tuple in itertools.combinations(positions, k):
        # for each chosen position, build the list of possible alternative bases
        # (i.e. anything except the original base at that position)
        choices_per_pos = [
            [b for b in alphabet if b != guide[p]]
            for p in pos_tuple
        ]
        # Cartesian product over those choices → one mutated guide per combo
        for repl_tuple in itertools.product(*choices_per_pos):
            guide_list = list(guide)
            for p, new_b in zip(pos_tuple, repl_tuple):
                guide_list[p] = new_b
            yield "".join(guide_list)

def chunked(iterable, size):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf

cluster_idx = 1
for cluster in clusters:
    for num_mismatches in range(1, max_mismatches+1):
        cluster_start = mm_coordinates[cluster[0]-1][1]
        cluster_end = mm_coordinates[cluster[-1]-1][2]
        print(f"total nuber of intended mismatches: {num_mismatches}")
        print(f"Cluster {cluster} starts at {cluster_start} and ends at {cluster_end}")
        print(f"Target sequences: {[target_name[i] for i in cluster]}")
        print("--------------------------------")
        search_start = cluster_start - scan_size
        search_end = cluster_end + scan_size
        print(f"Search region starts at {search_start} and ends at {search_end}")
        cluster_incl_WT = [0] + cluster # put the wt sequence to the head of list
        seqs = [target_seqs[cluster_idx].upper() for cluster_idx in cluster_incl_WT]
        selected_crRNAs = {target_name[i]: [] for i in cluster_incl_WT}

        for start_pos in range(search_start-28, search_end-28): # -28 because of crRNA spacer length
            for interest in range(len(seqs)):
                seq_of_interest = seqs[interest]
                idx_of_interest = cluster_incl_WT[interest]
                start_guide = [seq_of_interest[start_pos-1:start_pos-1+28]]


                guide_set = generate_intentional_mismatches(start_guide[0], num_mismatches = num_mismatches)

                # The predictive models require 10 nt of context on each side of the 28 nt probe-binding region
                seqs_frag = [s[start_pos - context_nt-1:start_pos - context_nt-1 + 48] for s in seqs]
                seqs_onehot = [prep_seqs.one_hot_encode(x) for x in seqs_frag]
                for candidate_guides in chunked(guide_set, BATCH_SIZE):
                    candidate_guides_onehot = [prep_seqs.one_hot_encode(g) for g in candidate_guides]
                    pred_perf_seqs, pred_act_seqs = cas13_cnn.run_full_model(candidate_guides_onehot, seqs_onehot)
                    weighted_perf_seqs = tf.math.subtract(tf.math.multiply(pred_act_seqs, tf.math.add(pred_perf_seqs, pos4)), pos4)
                    weighted_perf_seqs_np = weighted_perf_seqs.numpy()
                    # score
                    # instead of just subtracting the mean off-target from the on-target,
                    # put the additional penalty of top_off_target.
                    # later, I can add some weights to increase either of penalty.
                    on_target_act = weighted_perf_seqs_np[:,interest]
                    # threshold mask
                    cut_off_mask = on_target_act > cut_off
                    filtered_guide_set = np.array(candidate_guides)[cut_off_mask]
                    filtered_on_target_act = on_target_act[cut_off_mask]
                    filtered_perf = weighted_perf_seqs_np[cut_off_mask]

                    # top_off_target_activity
                    mask_for_top_off_target = filtered_perf != filtered_on_target_act[:, None]
                    arr = filtered_perf.copy()
                    arr[~mask_for_top_off_target] = np.nan
                    top_off_target_act = np.nanmax(arr, axis = 1)
                    mean_off_target_act = np.nanmean(arr, axis =1)
                    temp_score = filtered_on_target_act - (top_off_target_act + mean_off_target_act)/2
                    for ii in range(len(filtered_guide_set)):
                        selected_crRNAs[target_name[idx_of_interest]].append((start_pos, filtered_guide_set[ii], temp_score[ii], filtered_on_target_act[ii]))

                    selected_crRNAs[target_name[idx_of_interest]] = sorted(selected_crRNAs[target_name[idx_of_interest]], key=lambda x: -1e16 if np.isnan(x[2]) else x[2], reverse=True)[:top_k_crRNA]

        # make a dataframe to export
        rows = []
        for on_target_name, entries in selected_crRNAs.items():
            for pos, seq_list, score, on_target_activity in entries:
                rows.append({
                    "on_target_name": on_target_name,
                    "start_pos": pos,
                    "guide_sequence": seq_list,
                    "score": score,
                    "mean_on_target_act": on_target_activity,
                })

        final_output = pd.DataFrame(rows)
        os.makedirs("./output", exist_ok=True)
        final_output.to_excel(f"./output/best_crRNA_cluster_{cluster_idx}_mm_{num_mismatches}.xlsx")

    cluster_idx += 1

