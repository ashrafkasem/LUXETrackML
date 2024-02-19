#import tools
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
#internal
import os, sys, glob, yaml, datetime, argparse
import csv
import tensorflow as tf

Graph = namedtuple('Graph', ['X', 'Ri', 'Ro', 'y'])

class GraphDataset():
    def __init__(self, input_dir, n_samples=None):
        input_dir = os.path.expandvars(input_dir)
        filenames = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                     if f.endswith('.npz')]
        self.filenames = (
            filenames[:n_samples] if n_samples is not None else filenames)

    def __getitem__(self, index):
        return load_graph(self.filenames[index])

    def __len__(self):
        return len(self.filenames)

def get_dataset(input_dir,n_files):
    return GraphDataset(input_dir, n_files)
def load_graph(filename):
    """Reade a single graph NPZ"""
    '''
    with np.load(filename) as f:
        return sparse_to_graph(**dict(f.items()))
    '''
    return np.load(filename, allow_pickle=True)['arr_0']

def sparse_to_graph(X, Ri_rows, Ri_cols, Ro_rows, Ro_cols, y, dtype=np.float32):
    n_nodes, n_edges = X.shape[0], Ri_rows.shape[0]
    Ri_indices = np.vstack((Ri_rows, Ri_cols)).T
    Ro_indices = np.vstack((Ro_rows, Ro_cols)).T
    Ri = tf.sparse.SparseTensor(indices=Ri_indices, values=[dtype(1.0) for i in range(n_edges)], dense_shape=[n_nodes, n_edges])
    Ro = tf.sparse.SparseTensor(indices=Ro_indices, values=[dtype(1.0) for i in range(n_edges)], dense_shape=[n_nodes, n_edges])
    return Graph(X, Ri, Ro, y)

def parse_args():
    # generic parser, nothing fancy here
    parser = argparse.ArgumentParser(description='Load config file!')
    add_arg = parser.add_argument
    add_arg('config')
    add_arg('RID')
    return parser.parse_args()

def load_config_infer(args):
    with open(args.config, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
        config['run_number'] = 0
    return config

def load_config(args):
    # read the config file 
    with open(args.config, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
        if len(glob.glob(config['log_dir']))==0:
            os.mkdir(config['log_dir'])
        # append RID to log dir
        config['log_dir'] = config['log_dir']+'run{}/'.format(args.RID)
        if len(glob.glob(config['log_dir']))==0:
            os.mkdir(config['log_dir'])
        # print all configs
        print('Printing configs: ')
        for key in config:
            print(key + ': ' + str(config[key]))
        print('Log dir: ' + config['log_dir'])
        print('Training data input dir: ' + config['train_dir'])
        print('Validation data input dir: ' + config['valid_dir'])
        if config['run_type'] == 'new_run':
            delete_all_logs(config['log_dir'])
    # LOG the config every time
    with open(config['log_dir'] + 'config.yaml', 'w') as f:
        for key in config:
            f.write('%s : %s \n' %(key,str(config[key])))
    # return the config dictionary
    return config

def delete_all_logs(log_dir):
# Delete all .csv files in directory
    log_list = os.listdir(log_dir)
    for item in log_list:
        if item.endswith('.csv'):
            os.remove(log_dir+item)
            print(str(datetime.datetime.now()) + ' Deleted old log: ' + log_dir+item)
    init_all_logs(log_dir)

def init_all_logs(log_dir):
    with open(log_dir+'log_validation.csv', 'a') as f: 
        f.write('accuracy,auc,loss,precision,accuracy_3,precision_3,recall_3,f1_3,accuracy_5,precision_5,recall_5,f1_5,accuracy_7,precision_7,recall_7,f1_7,duration\n')
    with open(log_dir+'log_training.csv', 'a') as f: 
        f.write('accuracy,auc,loss,precision,accuracy_3,precision_3,recall_3,f1_3,accuracy_5,precision_5,recall_5,f1_5,accuracy_7,precision_7,recall_7,f1_7,duration\n')
    with open(log_dir+'summary.csv', 'a') as f:
        f.write('epoch,batch,loss,duration\n')


def log_parameters(log_dir, model, epoch):
    os.makedirs(log_dir+f"/check_points/tfmodel_check_point_epoch_{epoch}", exist_ok=True)
    model.save(log_dir+f"/check_points/tfmodel_check_point_epoch_{epoch}", save_format='tf') 

def log_gradients(log_dir, gradients):
    for idx, grads in enumerate(gradients):
        grads = grads.numpy().flatten()
        with open(log_dir+'log_gradients_%d.csv'%idx, 'a') as f:
            for idy, item in enumerate(grads):
                f.write('%f'%item  )
                if (idy+1)!=len(grads):
                    f.write(', ')
            f.write('\n')

def true_fake_weights(labels, dataset):
    '''
    [weight of fake edges, weight of true edges]
    '''
    if dataset == 'LUXE-TRACKING-5Perc':
        weight_list = [0.6838555406023465, 1.8597632096424799]
    elif dataset == 'LUXE-TRACKING-10Perc':
        weight_list = [0.5919166663106451, 3.2198549516046455]
    elif dataset == 'LUXE-TRACKING-20Perc':
        weight_list = [0.5459500700737486, 5.940688112091169]
    elif dataset == 'LUXE-TRACKING-30Perc':
        weight_list = [0.530632433113803, 8.66128444878085]
    elif dataset == 'LUXE-TRACKING-40Perc':
        weight_list = [0.5229721269815637, 11.382753704114448]
    else:
        raise ValueError('dataset not defined')
    return [weight_list[int(labels[i])] for i in range(labels.shape[0])]

def load_params(model, log_path):
    n_epochs = len(glob.glob(f'{log_path}/check_points/tfmodel_check_point_epoch_*'))
    last_epoch = n_epochs-1
    model_dir = f'{log_path}/check_points/tfmodel_check_point_epoch_{last_epoch}'
    loaded_model = tf.keras.models.load_model(model_dir)
    for idx, params in enumerate(loaded_model.trainable_variables):
            model.trainable_variables[idx].assign(params)
    return model, last_epoch+1
    
def get_configs(log_path_list):
    configs_dict = {}
    for path in log_path_list:
        # read the config file 
        n_runs = get_n_runs(path)
        path_ = path + 'run1/config.yaml'
        with open(path_, 'r') as ymlfile:
            config = yaml.load(ymlfile, Loader=yaml.FullLoader)
        # append n_runs to config
        config['n_runs'] = n_runs
        # save config to configs_dict
        configs_dict[path] = config

    # return the configs dictionary
    return configs_dict

def get_n_runs(path):
    n_folders = 0
    for _, dirnames, _ in os.walk(path):
        n_folders += len(dirnames)
    return n_folders
