import numpy as np
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf
from core.policies.graph_policy import Graph

def construct_graphs(hits, segments, feature_names=['h_x','h_y','h_z'], feature_scale=1):
    # Prepare the graph matrices
    n_hits = hits.shape[0]
    n_edges = segments.shape[0]
    scaler = preprocessing.StandardScaler()
    hits[feature_names] = scaler.fit_transform(hits[feature_names])
    X = (hits[feature_names].values / feature_scale).astype(np.float32)
    #Ri = np.zeros((n_hits, n_edges), dtype=np.uint8)
    #Ro = np.zeros((n_hits, n_edges), dtype=np.uint8)
    y = np.zeros(n_edges, dtype=np.float32)

    # We have the segments' hits given by dataframe label,
    # so we need to translate into positional indices.
    # Use a series to map hit label-index onto positional-index.
    hit_idx = pd.Series(np.arange(n_hits), index=hits.hit_id)
    seg_start = hit_idx.loc[segments.hit_id_0].values
    seg_end = hit_idx.loc[segments.hit_id_1].values

    # Now we can fill the association matrices.
    # Note that Ri maps hits onto their incoming edges,
    # which are actually segment endings.
    #Ri[seg_end, np.arange(n_edges)] = 1
    #Ro[seg_start, np.arange(n_edges)] = 1
    
    Ri_rows = seg_end
    Ri_cols = np.arange(n_edges)
    Ro_rows = seg_start
    Ro_cols = np.arange(n_edges)

    Ri_indices = np.vstack((Ri_rows, Ri_cols)).T
    Ro_indices = np.vstack((Ro_rows, Ro_cols)).T
    Ri = tf.sparse.SparseTensor(indices=Ri_indices, values=[1.0 for i in range(n_edges)], dense_shape=[n_hits, n_edges])
    Ro = tf.sparse.SparseTensor(indices=Ro_indices, values=[1.0 for i in range(n_edges)], dense_shape=[n_hits, n_edges])

    # Fill the segment labels
    y = segments['label']
    # Return a tuple of the results
    return Graph(X, Ri, Ro, y)