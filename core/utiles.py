import os 
import numpy as np

def getFileList(path,start_event,end_event, pattern = "e0gpc_7.0_%04d_positrons_edm4hep.tar.gz"):
    FileList = []

    #return nothing if path is a file
    if os.path.isfile(path):
        return []

    FileList = sorted([os.path.join(path,pattern) % i for i in range(start_event,end_event+1)])

    return FileList

def mean_across_dict_keys(dictionary):
    # Check if the dictionary is not empty
    if not dictionary:
        return []

    # Extract the values from the dictionary
    arrays = list(dictionary.values())

    # Stack the arrays along a new axis to facilitate mean calculation
    stacked_arrays = np.stack(arrays, axis=-1)

    # Calculate the mean along the last axis (axis=-1)
    mean_values = np.mean(stacked_arrays, axis=-1)

    # Convert the result to a list
    result_list = mean_values.tolist()

    return result_list


def pad_to_max_length(arrays):
    max_length = max(len(arr) for arr in arrays)
    padded_arrays = [np.pad(arr, (0, max_length - len(arr)), mode='constant', constant_values=np.nan) for arr in arrays]
    return padded_arrays

def summary_statistics_across_dict_keys(dictionary):
    # Check if the dictionary is not empty
    if not dictionary:
        return {'mean': [], 'min': [], 'max': []}

    # Extract the values from the dictionary
    arrays = list(dictionary.values())

    # Pad arrays to the maximum length
    padded_arrays = pad_to_max_length(arrays)

    # Use broadcasting for mean, min, and max
    mean_values = np.nanmean(padded_arrays, axis=0)
    min_values = np.nanmin(padded_arrays, axis=0)
    max_values = np.nanmax(padded_arrays, axis=0)

    # Convert the results to lists
    result = {'mean': mean_values.tolist(), 'min': min_values.tolist(), 'max': max_values.tolist()}

    return result
