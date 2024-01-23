import os
# Turn off warnings and errors due to TF libraries
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
import tensorflow as tf
from core.tools import *
import pandas as pd

def get_best_run(log_dir):
    nRuns = 5
    best_run = 0
    best_cutoff = 0.5
    auc_list = np.zeros(nRuns)
    # choose the best run based on final training stats
    for i in range(10):
        try:
            log = pd.read_csv(log_dir+'run%d/log_training.csv'%i)
            auc_list[i] = log['auc'].to_numpy()[-1]
        except:
            pass
    best_run = np.argmax(auc_list)
    # choose the best cutoff from the best run
    best_log = pd.read_csv(log_dir+'run%d/log_training.csv'%best_run)
    cutoff_list = [0.3,0.5,0.7]
    f1_cutoff = [best_log['f1_3'].to_numpy()[-1], best_log['f1_5'].to_numpy()[-1], best_log['f1_7'].to_numpy()[-1]]
    best_cutoff = cutoff_list[np.argmax(f1_cutoff)]
    print('Using run %d with cutoff %.1f'%(best_run, best_cutoff))
    return best_run, best_cutoff

if __name__ == '__main__':
    # Read config file
    config = load_config_infer(parse_args())
    # tools.config = config

    # Set GPU variables
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu']
    USE_GPU = (config['gpu']  != '-1')

    # Set number of thread to be used
    os.environ['OMP_NUM_THREADS'] = str(config['n_thread'])  # set num workers
    tf.config.threading.set_intra_op_parallelism_threads(config['n_thread'])
    tf.config.threading.set_inter_op_parallelism_threads(config['n_thread'])

    # Load the network
    from core.policies.gnn_policy import GNN
    GNN.config = config

    # setup model
    model = GNN()

    # load data
    datasets = get_dataset(config['inference_dir'], config['n_inference'])

    config['best_run'], _ = get_best_run(config['run_dir'])
    config['run_dir'] = f"{config['run_dir']}/run{config['best_run']}/"
    model, _ = load_params(model, config['run_dir'])
    print(model.summary())

    for i, data in enumerate(datasets): 
        X, Ri, Ro, y = data
        pred = model([X, Ri, Ro])
        file_name = f"preds_{datasets.filenames[i].split('/')[-1].replace('.npz','.npy')}"
        np.save(f"{config['output_dir']}/{file_name}", pred.numpy())
