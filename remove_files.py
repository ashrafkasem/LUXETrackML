import os
import glob
import sys
# Define the directory containing the files
directory = sys.argv[1]
numper_to_keep = sys.argv[2]
# Iterate over all files in the directory
for filename in os.listdir(directory):
    # Check if the file matches the pattern 'graph_*_split_*.npz'
    if filename.startswith('graph_') and filename.endswith('.npz'):
        # Extract the graph number (e.g., '146') from the filename
        graph_number = filename.split('_')[1]

        # Count the number of files with the same graph number
        files_with_graph_number = glob.glob(os.path.join(directory, f'graph_{graph_number}_split_*.npz'))
        num_files_with_graph_number = len(files_with_graph_number)

        # If there are more than 10 files with the same graph number, remove the excess files
        if num_files_with_graph_number > int(numper_to_keep):
            # Sort the files based on the split number
            files_with_graph_number.sort()

            # Remove the excess files (keeping the first 10)
            for file_to_remove in files_with_graph_number[10:]:
                os.remove(file_to_remove)
