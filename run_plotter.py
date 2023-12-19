import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# Script to compare different curves
'''
path0 = 'logs/01-05-2022-500HE/7.0_S/'
path1 = 'logs/01-05-2022-500HE-h128-CGNN-lr1e-2/7.0_S/'
path2 = 'logs/01-05-2022-500HE-h10-CGNN-lr1e-2/7.0_S/'
path3 = 'logs/01-05-2022-500HE-QGNN-h10-n10-l2/7.0_S/'
log_def = [path0, path1, path2, path3]
'''
# path0 = 'logs/01-05-2022-1000HE/7.0_S/'
# path1 = 'logs/01-05-2022-1000HE-h10-CGNN-lr1e-2/7.0_S/'
# path2 = 'logs/01-05-2022-1000HE-QGNN-h10-n10-l2/7.0_S/'
# log_def = [path0, path1, path2]
log_def = ["GNN_training_sampling_2/"]
path    = 'gnn-plots/bootstrapping_20_2/'
pngPath = path+'/png/'
pdfPath = path+'/pdf/'
epsPath = path+'/eps/'

if not os.path.exists(pngPath):
    os.makedirs(pngPath)
if not os.path.exists(pdfPath):
    os.makedirs(pdfPath)
if not os.path.exists(epsPath):
    os.makedirs(epsPath)

nRuns = 1
nEpochs = 100
#dataType = 'validation'
dataType = 'training'
metrics_2_plot = ['auc','loss','precision','accuracy_3','precision_3','recall_3','accuracy_5','precision_5','recall_5','accuracy_7','precision_7','recall_7','f1_7']
#prepare logs
log_labels = [
        'GNN - D=128, it=4, lr=1e-3, bsize=1',
        'GNN - D=128, it=4, lr=1e-3, bsize=5',
        # 'CGNN - D=10, it=4, lr=1e-2',
        # 'QGNN - D=10, it=4, N=10, L=2, lr=1e-2',
        ]
loss_dict = {}
log_list = []
for i in range(len(log_def)):
    log_list.append(log_def[i])
    loss_dict[log_list[i]] = np.zeros((nRuns,nEpochs+1)) * np.nan

for metric in metrics_2_plot:
    plt.clf()
    plt.figure()
    for idx, path in enumerate(log_list):
        for idy in range(nRuns):
            df = pd.read_csv(path+'run%d/log_'%(idy)+dataType+'.csv')
            if len(df[metric]) < nEpochs:
                if metric=='auc':
                    loss_dict[path][idy][:len(df[metric])] = 1-df[metric]
                else:
                    loss_dict[path][idy][:len(df[metric])] = df[metric]
            else:
                if metric=='auc':
                    loss_dict[path][idy] = 1-df[metric][:nEpochs+1]
                else:
                    loss_dict[path][idy] = df[metric][:nEpochs+1]
    
        plt.plot(np.nanmean(loss_dict[path], axis=0), label=log_labels[idx])
        plt.fill_between(
            range(nEpochs+1),
            np.nanmin(loss_dict[path], axis=0),
            np.nanmax(loss_dict[path], axis=0),
            alpha=0.2
            )

    plt.title('data:2023-Xi-7.0 - GNN')
    plt.xlabel('epochs')
    if metric=='auc':
        metric_ = '1-AUC (lower the better)'
        plt.yscale('log')
        plt.ylim([1e-6, 1.0])
    elif metric=='loss':
        metric_ = 'Loss (lower the better)'
        plt.yscale('log')
    else:
        metric_=metric
    plt.ylabel(metric_)
    #plt.ylim([0.2, 1.0])
    plt.legend()

    plt.savefig(pngPath+dataType+metric+'.png', dpi=600)
    #plt.savefig(pdfPath+'classical_'+metric+'.pdf')
    #plt.savefig(epsPath+'classical_'+metric+'.eps')
    plt.close()