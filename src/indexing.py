import numpy as np

def get_word_indices(word_list, index):
    common_indices = []
    new_word_list = []
    for word in word_list:
        try:
            common_indices.append(index[word])
            new_word_list.append(word)
        except KeyError:
            print("Unmapped word!")
    return new_word_list, np.array(common_indices)