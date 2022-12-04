import os
import click
from pathlib import Path

import numpy as np
from collections import Counter
from multiprocessing import Queue, Process
from queue import Empty

import pickle as pickle
from ioutils import load_pickle, mkdir
from utils import process_lemma_line

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
from sparse_io import export_mat_from_dict

def worker(proc_num, queue, window_size, type, id_map, data_dir, out_dir):
    while True:
        try:
            decade = str(queue.get(block=False))
        except Empty:
             break
        print("Proc:", proc_num, "Decade:", decade)
        pair_counts = Counter()
        for file in os.listdir(os.path.join(data_dir, decade)):
            with open(Path(data_dir, decade, file)) as fp:
                print(proc_num, file)
                fp.readline()
                context = []
                for line in fp:
                    word, lemma, lemma_pos, _ = process_lemma_line(line)
                    if type == "word":
                        item = word
                    elif type == "lemma":
                        item = lemma
                    elif type == "lemma_pos":
                        item = lemma_pos
                    else:
                        raise Exception("Unknown type {}".format(type))
                    if item == None:
                        continue
                    context.append(id_map[item])
                    if len(context) > window_size * 2 + 1:
                        context.pop(0)
                    pair_counts = _process_context(context, pair_counts, window_size)
        decade += ".bin"
        export_mat_from_dict(pair_counts, str(out_dir / decade).encode('utf-8'))

def _process_context(context, pair_counts, window_size):
    if len(context) < window_size + 1:
        return pair_counts
    target = context[window_size]
    indices = list(range(0, window_size))
    indices.extend(range(window_size + 1, 2 * window_size + 1))
    for i in indices:
        if i >= len(context):
            break
        pair_counts[(target, context[i])] += 1
    return pair_counts

@click.command()
@click.option(
    '--data', 
    type=click.Path(exists=True), 
    required=True, 
    default="data", 
    help="Path to data directory"
    )
@click.option(
    '--info', 
    type=click.Path(exists=True), 
    required=True, 
    default="info", 
    help="Path to info directory"
    )
@click.option(
    '--type',
    type=click.Choice(["word", "lemma", "lemma_pos"]),
    required=True,
    default="word",
    help="Type of cooccurrence matrix to build"
    )
@click.option(
    '--window-size',
    type=int,
    required=True,
    default=4,
    help="Window size for cooccurrence matrix"
    )
@click.option(
    '--out',
    type=click.Path(),
    required=True,
    default=Path("cooccurs"),
    help="Path to output directory"
    )
@click.option(
    '--workers',
    type=int,
    required=True,
    default=25,
    help="Number of workers"
    )
@click.option(
    '--start', 
    type=int, 
    required=True, 
    default=1810, 
    help="Start year"
    )
@click.option(
    '--end', 
    type=int, 
    required=True, 
    default=2010, 
    help="End year"
    )
@click.option(
    '--step', 
    type=int, 
    required=True, 
    default=10, 
    help="End year"
    )
def main(data, info, type, window_size, workers, out, start, end, step):
    path_dict = Path(info) / "{type}-dict.pkl".format(type=type)
    out_dir = Path(out) / "{type}/{window_size:d}/".format(type=type, window_size=window_size)
    if not out_dir.exists():
        mkdir(out_dir)

    queue = Queue()
    for decade in range(start, end, step):
        queue.put(decade)
    id_map = load_pickle(path_dict)
    procs = [
        Process(
        target=worker, 
        args=[i, queue, window_size, type, id_map, data, out_dir]) for i in range(workers)
        ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

if __name__ == "__main__":
    main()
