import sys
import os
# Turn off warnings and errors due to TF libraries
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
import time
import datetime
import csv
from random import shuffle
import tensorflow as tf
# import internal scripts
from core.tools import *
from core.test import calculate_metrics, validate
###############################################################################
def batch_train_step(n_step):
    '''combines multiple  graph inputs and executes a step on their mean'''
    with tf.GradientTape() as tape:
        for batch in range(config['batch_size']):
            X, Ri, Ro, y = train_data[
                train_list[n_step*config['batch_size']+batch]
                ]
            y=y.to_numpy()
            label = tf.reshape(tf.convert_to_tensor(y),shape=(y.shape[0],1))
            
            if batch==0:
                # calculate weight for each edge to avoid class imbalance
                weights = tf.convert_to_tensor(true_fake_weights(y, config['dataset']))
                # reshape weights
                weights = tf.reshape(tf.convert_to_tensor(weights),
                                     shape=(weights.shape[0],1))
                preds = model([X,Ri,Ro])
                labels = label
            else:
                weight = tf.convert_to_tensor(true_fake_weights(y, config['dataset']))
                # reshape weights
                weight = tf.reshape(tf.convert_to_tensor(weight),
                                    shape=(weight.shape[0],1))

                weights = tf.concat([weights, weight],axis=0)
                preds = tf.concat([preds, model([X,Ri,Ro])],axis=0)
                labels = tf.concat([labels, label],axis=0)

        loss_eval = loss_fn(labels, preds, sample_weight=weights)

    grads = tape.gradient(loss_eval, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    
    batch_history = calculate_metrics(labels,preds)    

    return loss_eval, grads, batch_history

def model_save():
    pass

def model_load():
    pass

if __name__ == '__main__':
    # Read config file
    config = load_config(parse_args())
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
    train_data = get_dataset(config['train_dir'], None) #config['n_train'])
    train_list = [i for i in range(len(train_data))]

    # execute the model on an example data to test things
    X, Ri, Ro, y = train_data[0]
    model([X, Ri, Ro])

    # print model summary
    print(model.summary())

    # Log initial parameters if new run
    if config['run_type'] == 'new_run':    
        if config['log_verbosity']>=2:
            log_parameters(config['log_dir'], model.trainable_variables)
        epoch_start = 0

        # Test the validation set
        if config['n_valid']: validate(config, model)
        else: 
            raise ValueError('n_valid is not defined!')
    # Load old parameters if continuing run
    elif config['run_type'] == 'continue':
        # load params 
        model, epoch_start = load_params(model, config['log_dir'])
    else:
        raise ValueError('Run type not defined!')

    # Get loss function and optimizer
    loss_fn = getattr(tf.keras.losses, config['loss_func'])()
    opt = getattr(
        tf.keras.optimizers,
        config['optimizer'])(learning_rate=config['lr_c']
    )

    # Print final message before training
    if epoch_start == 0: 
        print(str(datetime.datetime.now()) + ': Training is starting!')
    else:
        print(
            str(datetime.datetime.now()) 
            + ': Training is continuing from epoch {}!'.format(epoch_start+1)
            )
    # Lists to store loss and accuracy values per epoch
    
    # Start training
    for epoch in range(epoch_start, config['n_epoch']):
        t_epoch0 = datetime.datetime.now()  
        shuffle(train_list) # shuffle the order every epoch
        # Initialize variables to accumulate metrics
        
        total_loss = 0.0
        total_duration = 0

        total_metrics = {
            'accuracy_5': 0.0,
            'auc': 0.0,
            'precision_5': 0.0,
            'accuracy_3': 0.0,
            'precision_3': 0.0,
            'recall_3': 0.0,
            'f1_3': 0.0,
            'accuracy_5': 0.0,
            'precision_5': 0.0,
            'recall_5': 0.0,
            'f1_5': 0.0,
            'accuracy_7': 0.0,
            'precision_7': 0.0,
            'recall_7': 0.0,
            'f1_7': 0.0,
            
        }
        for n_step in range(config['n_train']//config['batch_size']):
            # start timer
            t0 = datetime.datetime.now()  

            # iterate a step
            loss_eval, grads, metrics_dict = batch_train_step(n_step)
            total_loss += loss_eval
            # end timer
            dt = datetime.datetime.now() - t0  
            t = dt.seconds + dt.microseconds * 1e-6 # time spent in seconds
            
            for metric_name, metric_value in metrics_dict.items():
                total_metrics[metric_name] += metric_value

            # Print summary
            print(
                str(datetime.datetime.now())
                + ": Epoch: %d, Batch: %d, Loss: %.4f, Elapsed: %dm%ds" \
                %(epoch+1, n_step+1, loss_eval.numpy() ,t / 60, t % 60)
                )
            
            # Start logging 
            
            # Log summary 
            with open(config['log_dir']+'summary.csv', 'a') as f:
                f.write(
                    '%d, %d, %f, %f\n' \
                    %(epoch+1, n_step+1, loss_eval.numpy(), t)
                    )            
            # Test every TEST_every
            #if (n_step+1)%config['TEST_every']==0:
            # instead of testing every TEST_every we test every epoch
            # this can move to outside of the n_step loop but keep it for the moment
            # if (n_step+1)%(config['n_train']//config['batch_size'])==0:
            #     test(config, model, 'valid')
            #     test(config, model, 'train')
        t_epoch = datetime.datetime.now()  
        total_duration = t_epoch0 - t_epoch
        total_duration = total_duration.seconds + total_duration.microseconds * 1e-6 # time spent in seconds
        # save model checkpoints
        # Log parameters
        log_parameters(config['log_dir'], model, epoch)
        # Log gradients
        if config['log_verbosity']>=2:
            log_gradients(config['log_dir'], grads)

        avg_loss = total_loss / (config['n_train']//config['batch_size'])
        # Calculate average metrics for the epoch
        for metric_name in total_metrics:
            total_metrics[metric_name] /= (config['n_train']//config['batch_size'])

        # Log Metrics
        with open(config['log_dir']+'log_training.csv', 'a') as f:
            f.write('%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %d\n' % (
                total_metrics['accuracy_5'],
                total_metrics['auc'],
                avg_loss,
                total_metrics['precision_5'],
                total_metrics['accuracy_3'],
                total_metrics['precision_3'],
                total_metrics['recall_3'],
                total_metrics['f1_3'],
                total_metrics['accuracy_5'],
                total_metrics['precision_5'],
                total_metrics['recall_5'],
                total_metrics['f1_5'],
                total_metrics['accuracy_7'],
                total_metrics['precision_7'],
                total_metrics['recall_7'],
                total_metrics['f1_7'],
                total_duration
            ))
        # run on validation datasets
        validate(config, model)

    print(str(datetime.datetime.now()) + ': Training completed!')
    model.save(config['log_dir']+"/tfmodel", save_format='tf')