import os
import click
from pathlib import Path

from utils import process_lemma_line
from ioutils import write_pickle

def process_file(fp, word_dict, lemma_dict, lemma_pos_dict):
    fp.readline()
    for line in fp:
        word, lemma, lemma_pos, _ = process_lemma_line(line)
        if word == None:
            continue
        if lemma_pos == None:
            continue
        if word not in word_dict:
            id = len(word_dict)
            word_dict[word] = id
        if lemma not in lemma_dict:
            id = len(lemma_dict)
            lemma_dict[lemma] = id
        if lemma_pos not in lemma_pos_dict:
            id = len(lemma_pos_dict)
            lemma_pos_dict[lemma_pos] = id

@click.command()
@click.option(
    '--data', 
    type=click.Path(exists=True), 
    required=True, 
    default="data", 
    help="Path to data directory"
    )
@click.option(
    '--out', 
    type=click.Path(), 
    required=True, 
    default="info", 
    help="Path to output directory"
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
def main(data, out, start, end, step):
    word_dict = {}
    lemma_dict = {}
    lemma_pos_dict = {}
    for decade in range(start, end, step):
        folder = str(decade)
        print("Processing decade...", folder)
        for file in os.listdir(os.path.join(data, folder)):
            with open(Path(data, folder, file)) as fp:
                print("Processing file..", folder + "/" + file)
                process_file(fp, word_dict, lemma_dict, lemma_pos_dict)

    if not Path(out).exists():
        Path(out).mkdir(parents=True)

    write_pickle(word_dict, Path(out) / "word-dict.pkl") 
    write_pickle(lemma_dict, Path(out) / "lemma-dict.pkl") 
    write_pickle(lemma_pos_dict, Path(out) / "lemma-pos-dict.pkl")

if __name__ == "__main__":
    main()