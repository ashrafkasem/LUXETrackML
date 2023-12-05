import sys
sys.path.insert(0,'..')
import time
import tqdm
import numpy as np
import pandas as pd

from core.data.construct_graphs import construct_graphs

def construct_segments(event, cut_list, getGraph=False, getGraphandSegments=False):
    keys = ['hit_id', 'event', 'particle_id', 'h_layer', 'h_x', 'h_y', 'h_z']
    # event['h_layer'] = [int(layer[-1]) for layer in event['h_layer'].to_list()]
    doublets = []
    print(f"started making graphs for event number {np.unique(event['event'])} which has size of {event.shape}")
    for idx, row in event[event['h_layer']!=4].iterrows():
        hits0 = pd.DataFrame({k: [v] for k, v in dict(row).items()})
        layer_ID = hits0['h_layer'][0]#int(hits0['h_layer'])
        hits1 = event[event['h_layer']==(layer_ID+1)]
        hit_pairs = hits0[keys].reset_index().merge(
            hits1[keys].reset_index(), on='event', suffixes=('_0', '_1'))
        # Apply pre-selection to hit pairs
        selected_doublets = segment_selector(hit_pairs, cut_list)
        doublets.append(selected_doublets)
    doublets = pd.concat(doublets)
    if getGraph or getGraphandSegments:
        # Construct graph
        graph =  construct_graphs(event, doublets[doublets['selected']==1])
        if getGraphandSegments:
            return doublets, graph
        else:
            return graph
    else:
        return doublets

def segment_selector(segments, cut_list):
    x_cut, x_cut_epsilon, y_cut, z_value = cut_list
    x0, dx, dy  = get_coordinate_at_z_value(segments, z_value)
    selected = np.abs((dx)/x0 - x_cut) < x_cut_epsilon
    selected = np.logical_and(selected, dy/x0 < y_cut)

    label = segments.particle_id_0 == segments.particle_id_1
    segments['label']    = label.astype(int)
    segments['selected'] = selected.astype(int)
    # drop all not selected segments
    segments.drop(segments[segments.selected!= 1].index, inplace=True)
    return segments

def get_coordinate_at_z_value(segments, z_value):
    dx_p1_p2 = segments.h_x_1 - segments.h_x_0
    dy_p1_p2 = segments.h_y_1 - segments.h_y_0
    dz_p1_p2 = segments.h_z_1 - segments.h_z_0
    x_at_origin = segments.h_x_1 - dx_p1_p2 * np.abs(segments.h_z_1 - z_value) / dz_p1_p2
    x_at_origin[segments.h_z_0==z_value] = segments[segments.h_z_0==z_value].h_x_0
    x_at_origin[segments.h_z_1==z_value] = segments[segments.h_z_1==z_value].h_x_1
    return x_at_origin, dx_p1_p2, dy_p1_p2