"""
++ v4
optimization, jit compiler

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

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging
import natsort

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger()

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"          # stop pre-allocating the whole GPU
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"       # reduces fragmentation on CUDA>=11.2

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
tf.config.threading.set_inter_op_parallelism_threads(max(2, os.cpu_count()//2))
# JIT / XLA & device placement
tf.config.optimizer.set_jit(True)
tf.config.set_soft_device_placement(True)

from badgers_mod.utils import prepare_sequences as prep_seqs
from badgers_mod.utils import cas13_cnn as cas13_cnn



import itertools
from Bio import Align

# adapt prediction parameters (don't change unless necessary)
grid = {'c': 1.0, 'a': 3.769183, 'k': -3.833902, 'o': -2.134395, 't2w': 2.973052}
context_nt = 10
pos4 = tf.constant(4.0)

# search parameter
scan_size = 10
top_k_crRNA = 4
#max_mismatches = 5
#cut_off = -2
#BATCH_SIZE = 2**11
#project_name = "test"
#file_name = 'INH_inhA_targets_incl_WT.xlsx'
#is_cluster = True
parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch_size", help="batch size for crRNA search. change it based on the GPU memory", type=int, default=2048)
parser.add_argument("-c", "--cut_off", help="cut-off on-target activity", type=float, default=-2)
parser.add_argument("-p", "--project_name", help="project name", type=str, default="output")
parser.add_argument("-f", "--file_name", help="file name", type=str, default="INH_inhA_targets_incl_WT.xlsx")
parser.add_argument("-mm", "--max_mismatches", help="maximum mismatch", type=int, default=5)
parser.add_argument("-cl", "--is_cluster", help="are the input sequences already clustered?", type=bool, default=False)
args = parser.parse_args()

BATCH_SIZE = args.batch_size
cut_off = args.cut_off
project_name = args.project_name
file_name = args.file_name
max_mismatches = args.max_mismatches
is_cluster = args.is_cluster

output_dir = os.path.join("./output", project_name)
os.makedirs(output_dir, exist_ok=True)

# heatmap parameter
cell_size = 0.4  # in the heatmap (inch)
color_map_type = "YlGnBu"  # "viridis" # or "YlGnBu"
vmin = - 3.5
vmax = - 1

# gen_guide, target_set1, target_set2
#gen_guide_file = './cluster1-3_final_merge.xlsx'
target_list_file = os.path.join('./',file_name)
#gen_guide_data = pd.read_excel(gen_guide_file, header=0)

target_data = pd.read_excel(target_list_file, header=0)
target_seqs = target_data.new_seq.to_list()
WT_seq = target_seqs[0].upper() # change this in case that you have WT (or base seq) not in the first row.
target_name = target_data.new_name.to_list()

mm_coordinates = []
aligner = Align.PairwiseAligner()

# check where mismatch starts and ends
for i in range(len(target_seqs)-1):
    ii = i+1
    alignments = aligner.align(WT_seq, target_seqs[ii].upper())
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

if not is_cluster:
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
            current_cluster = natsort.natsorted(current_cluster)
            clusters.append(current_cluster)
            current_cluster = [idx]
            anchor_start = start

    # add the last cluster
    if current_cluster:
        current_cluster = natsort.natsorted(current_cluster)
        clusters.append(current_cluster)
else:
    clusters = [[i+1 for i in range(len(target_seqs)-1)]]

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
tf.config.optimizer.set_jit(True)  # XLA on

@tf.function(jit_compile=True)
def score_batch(guides_onehot, seqs_onehot, pos4 = pos4):
    # returns weighted [B, T]
    pred_perf, pred_act = cas13_cnn.run_full_model(guides_onehot, seqs_onehot)
    return pred_act * (pred_perf + pos4) - pos4

@tf.function(jit_compile=True)
def reduce_offtargets(weighted, interest):
    # weighted: [B, T] (float16/float32)
    T = tf.shape(weighted)[1]
    col_mask = tf.ones([T], dtype=tf.bool)
    col_mask = tf.tensor_scatter_nd_update(col_mask, indices=[[interest]], updates=[False])
    off = tf.boolean_mask(weighted, col_mask, axis=1)        # [B, T-1]
    on = weighted[:, interest]                                # [B]
    top_off = tf.reduce_max(off, axis=1)                      # [B]
    mean_off = tf.reduce_mean(off, axis=1)                    # [B]
    return on, top_off, mean_off

def guide_batches(start_seq, k, batch):
    def gen():
        for gids in chunked(generate_intentional_mismatches(start_seq, k), batch):
            arr = np.stack([prep_seqs.one_hot_encode(g) for g in gids], axis=0).astype('float32')
            yield arr, np.asarray(gids, dtype=np.str_)

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(None, 28, 4), dtype=tf.float32),  # guides one-hot
            tf.TensorSpec(shape=(None,), dtype=tf.string),  # raw guide strings
        ),
    ).prefetch(tf.data.AUTOTUNE)


cluster_idx = 1
for cluster in clusters:
    for num_mismatches in range(1, max_mismatches+1):
        cluster_start = mm_coordinates[cluster[0]-1][1]
        cluster_end = mm_coordinates[cluster[-1]-1][2]
        logger.info(f"total nuber of intended mismatches: {num_mismatches}")
        logger.info(f"Cluster {cluster} starts at {cluster_start} and ends at {cluster_end}")
        logger.info(f"Target sequences: {[target_name[i] for i in cluster]}")
        logger.info("--------------------------------")
        search_start = cluster_start - scan_size
        search_end = cluster_end + scan_size
        logger.info(f"Search region starts at {search_start} and ends at {search_end}")
        cluster_incl_WT = [0] + cluster # put the wt sequence to the head of list
        seqs = [target_seqs[cluster_idx].upper() for cluster_idx in cluster_incl_WT]
        selected_crRNAs = {target_name[i]: [] for i in cluster_incl_WT}

        for start_pos in range(search_start-28, search_end-28): # -28 because of crRNA spacer length
            # The predictive models require 10 nt of context on each side of the 28 nt probe-binding region
            seqs_frag = [s[start_pos - context_nt - 1:start_pos - context_nt - 1 + 48] for s in seqs]
            seqs_onehot_np = np.stack([prep_seqs.one_hot_encode(x) for x in seqs_frag], axis=0)
            seqs_onehot = tf.convert_to_tensor(seqs_onehot_np, dtype=tf.float32)

            for interest in range(len(seqs)):
                seq_of_interest = seqs[interest]
                idx_of_interest = cluster_incl_WT[interest]
                start_guide = [seq_of_interest[start_pos-1:start_pos-1+28]]

                for guides, raw_guides in guide_batches(start_guide[0], num_mismatches, BATCH_SIZE):
                    weighted = score_batch(guides, seqs_onehot)

                    on, top_off, mean_off = reduce_offtargets(weighted, interest)
                    keep_mask = on > cut_off

                    # compute score on GPU, then pull back only the small kept slice
                    temp_score = on - (top_off + mean_off)
                    idx_keep = tf.where(keep_mask)[:, 0]
                    if tf.size(idx_keep) == 0:
                        continue

                    kept_scores = tf.gather(temp_score, idx_keep).numpy().astype("float32")
                    kept_on = tf.gather(on, idx_keep).numpy().astype("float32")
                    # gather raw strings; result is tf.string -> bytes in numpy, decode to str
                    kept_guides = tf.gather(raw_guides, idx_keep).numpy()
                    kept_guides = [g.decode("utf-8") for g in kept_guides]

                    tgt_key = target_name[idx_of_interest]
                    bucket = selected_crRNAs[tgt_key]
                    for seq_, sc_, on_ in zip(kept_guides, kept_scores, kept_on):
                        bucket.append((start_pos, seq_, sc_, on_))
                    # keep only top_k per on-target
                    bucket.sort(key=lambda x: (-1e16 if np.isnan(x[2]) else x[2]), reverse=True)
                    del bucket[top_k_crRNA:]

                    selected_crRNAs[tgt_key] = bucket

        # make a dataframe to export
        rows = []
        fig_data = []
        for on_target_name, entries in selected_crRNAs.items():
            for pos, seq_list, score, on_target_activity in entries:
                rows.append({
                    "on_target_name": on_target_name,
                    "start_pos": pos,
                    "guide_sequence": seq_list,
                    "score": score,
                    "mean_on_target_act": on_target_activity,
                })

                # plotting
                seqs_fig = [s.upper()[pos - context_nt - 1:pos - context_nt - 1 + 48] for s in seqs]
                seqs_fig_onehot_np = np.stack([prep_seqs.one_hot_encode(x) for x in seqs_fig], axis=0)
                seqs_fig_onehot = tf.convert_to_tensor(seqs_fig_onehot_np, dtype=tf.float32)

                plotting_guides = tf.convert_to_tensor([prep_seqs.one_hot_encode(seq_list)], dtype=tf.float32)
                weighted_fig = score_batch(plotting_guides, seqs_fig_onehot)
                fig_data.append(weighted_fig)

        final_output = pd.DataFrame(rows)
        final_output_name = os.path.join(output_dir,f"./best_crRNA_co_{cut_off}_cluster_{cluster_idx}_mm_{num_mismatches}")
        final_output.to_excel(final_output_name + ".xlsx")

        # for plot
        final_fig_data = np.array(fig_data).reshape(len(fig_data), -1)
        fig, ax = plt.subplots(figsize=(cell_size * final_fig_data.shape[0] + 4,
                                        cell_size * final_fig_data.shape[1] + 1),
                               dpi=300)

        sns.heatmap(final_fig_data.transpose(), cmap=color_map_type,
                    square=True,
                    ax=ax, linewidth=0.3, linecolor='black',
                    vmin=vmin, vmax=vmax,
                    yticklabels = [target_name[i] for i in cluster_incl_WT])
        plt.show()
        fig.savefig(final_output_name + "_score.png", bbox_inches='tight', dpi=300)



    cluster_idx += 1








