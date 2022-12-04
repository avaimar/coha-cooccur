import numpy as np
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
from sparse_io import retrieve_mat_as_coo

def load_matrix(f):
    print(f)
    if not f.endswith('.bin'):
        f += ".bin"
    # froom sparse_io import retrieve_mat_as_coo
    return retrieve_mat_as_coo(f.encode('utf-8')).tocsr()