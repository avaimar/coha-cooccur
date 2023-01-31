import numpy as np
import os
from gensim.models import KeyedVectors
import glob


def present_word(model, word):
    return word in model and np.linalg.norm(model.vectors[model.key_to_index[word]]) > 1e-6


def compute_mean_vector(model, words, unique=False):
    if unique:
        words = list(set(words))
    nonzero_w = [w for w in words if present_word(model, w)]
    if len(nonzero_w) == 0:
        return None
    mean_vec = model.get_mean_vector(nonzero_w, pre_normalize=False, ignore_missing=True, post_normalize=True)
    return mean_vec


def compute_bias_score(attribute_vecs, t1_mean, t2_mean, cosine=False):
    t2_component = attribute_vecs - t2_mean.reshape(1, -1)
    t2_component = np.linalg.norm(t2_component, axis=1)

    t1_component = attribute_vecs - t1_mean.reshape(1, -1)
    t1_component = np.linalg.norm(t1_component, axis=1)
    bias_score = np.mean(t1_component - t2_component)

    if cosine:
        t1_component = np.matmul(attribute_vecs, t1_mean.reshape(-1, 1))
        t2_component = np.matmul(attribute_vecs, t2_mean.reshape(-1, 1))
        bias_score = np.mean(t2_component - t1_component)
    return bias_score, t1_component, t2_component


def load_coha():
    vectors_list = glob.glob('../../Replication-Garg-2018/data/coha-word/*vectors.txt')
    vectors = {}
    for file_name in vectors_list:
        file_decade = file_name.split(os.path.sep)[-1][:4]
        vectors[file_decade] = KeyedVectors.load_word2vec_format(file_name, binary=False, no_header=True)
    return vectors
