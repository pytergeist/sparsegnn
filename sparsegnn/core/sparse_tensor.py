# sparse_tensor.py

import tensorflow as tf
import numpy as np
from scipy.sparse import coo_matrix


def coo_to_sparse_tensor(coo: coo_matrix):
    """Convert a scipy.sparse.coo_matrix to a tf.SparseTensor.

    Args:
        coo (scipy.sparse.coo_matrix): The coo_matrix to convert.

    Returns:
        tf.SparseTensor: The converted sparse tensor.
    """
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(
        indices=indices,
        values=coo.data,
        dense_shape=coo.shape,
    )

# TO DO:
    # sparse_tensor_to_coo
    # sparse_tensor_to_dense
    # dense_to_sparse_tensor
    # coo_to_dense
    # dense_to_coo