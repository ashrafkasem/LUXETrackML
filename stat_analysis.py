import numpy as np
import glob
import sys

xi=7.0
data_dir = sys.argv[1]#'./output'
file_lists = sorted(glob.glob(f"{data_dir}/stats_*.npy"))

nParticles = 0
true_count = 0
fake_count = 0
true_generated_count = 0
for file_ in (file_lists):
    stats = np.load(file_)
    nParticles += stats[0]
    true_count += stats[1]
    fake_count += stats[2]
    true_generated_count += stats[3]

eff = true_count/(true_generated_count)
purity = true_count/(true_count+fake_count)
weight_list = [(1/(1-purity))/2, (1/purity)/2]
print('Graph Production Statistics......')
print('Using %d events, %d particles in total'%(len(file_lists), nParticles))
print('Total: True edges: ', true_count, ' Fake edges: ', fake_count)
print('Efficiency: {:.4f}, Purity: {:.4f}'.format(eff, purity))
print('Weights for classifier training: ', weight_list)
