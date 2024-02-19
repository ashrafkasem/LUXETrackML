import sys
import os
import time
import datetime
import numpy as np
from sklearn import metrics
from core.tools import *
import tensorflow as tf

def validate(config, model):
    print(
        str(datetime.datetime.now()) 
        + ' Starting testing the validation set with '
        + str(config['n_valid']) + ' subgraphs!'
        )

    # Start timer
    t_start = time.time()
    
    # load data
    valid_data = get_dataset(config['valid_dir'], config['n_valid'])
    n_test = config['n_valid']

    # obtain graphIDs from filenames
    gIDs = []
    for path in valid_data.filenames:
        gIDs.append(path.split('/')[-1].split('.')[0])

    # prepare directory to save predictions
    if len(glob.glob(config['log_dir']+'predictions/'))==0:
        os.mkdir(config['log_dir']+'predictions/')


    # Load loss function
    loss_fn = getattr(tf.keras.losses, config['loss_func'])()

    # Obtain predictions and labels
    for n in range(n_test):

        X, Ri, Ro, y = valid_data[n]
        y=y.to_numpy()
        if n == 0:
            preds = model([X, Ri, Ro])
            labels = y
            # save the prediction
            np.save(config['log_dir']+'predictions/preds_'+gIDs[n]+'.npy', preds.numpy())
        else:	
            out = model([X, Ri, Ro])
            preds  = tf.concat([preds, out], axis=0)
            labels = tf.concat([labels, y], axis=0)
            # save the prediction
            np.save(config['log_dir']+'predictions/preds_'+gIDs[n]+'.npy', out.numpy())

    labels = tf.reshape(labels, shape=(labels.shape[0],1))

    # calculate weight for each edge to avoid class imbalance
    weights = tf.convert_to_tensor(true_fake_weights(labels, config['dataset']))

    loss = loss_fn(labels, preds, sample_weight=weights).numpy()

    # Log all predictons (takes some considerable time - use only for debugging)
    if config['log_verbosity']>=3:	
        with open(config['log_dir']+'log_validation_preds.csv', 'a') as f:
            for i in range(len(preds)):
                f.write('%.4f, %.4f\n' %(preds[i],labels[i]))
    
    history = calculate_metrics(labels,preds)
    
    # End timer
    duration = time.time() - t_start
    
    # Log Metrics
    with open(config['log_dir']+'log_validation.csv', 'a') as f:
        f.write('%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %d\n' %(
            history['accuracy_5'],
            history['auc'],
            loss, 
            history['precision_5'],
            history['accuracy_3'], 
            history['precision_3'], 
            history['recall_3'], 
            history['f1_3'], 
            history['accuracy_5'], 
            history['precision_5'], 
            history['recall_5'], 
            history['f1_5'], 
            history['accuracy_7'], 
            history['precision_7'], 
            history['recall_7'], 
            history['f1_7'], 
            duration))

    # Print summary
    print(str(datetime.datetime.now()) + ': Valid Test:  Loss: %.4f,  AUC: %.4f, Acc: %.4f,  Precision: %.4f -- Elapsed: %dm%ds' %(
        loss, history['auc'], history['accuracy_5']*100, history['precision_5'], duration/60, duration%60))

    del labels
    del preds

def calculate_metrics(labels, preds):
    # Calculate Metrics
    # To Do: add 0.8 threshold and other possible metrics
    # efficency, purity etc.
    labels = labels.numpy()
    preds  = preds.numpy()

    #n_edges = labels.shape[0]
    #n_class = [n_edges - sum(labels), sum(labels)]

    fpr, tpr, _ = metrics.roc_curve(labels.astype(int),preds,pos_label=1 )
    auc                = metrics.auc(fpr,tpr)

    tn, fp, fn, tp = metrics.confusion_matrix(
        labels.astype(int),(preds > 0.3)*1
        ).ravel() # get the confusion matrix for 0.3 threshold
    accuracy_3  = (tp+tn)/(tn+fp+fn+tp)
    precision_3 = tp/(tp+fp) # also named purity
    recall_3    = tp/(tp+fn) # also named efficiency
    f1_3        = (2*precision_3*recall_3)/(precision_3+recall_3) 

    tn, fp, fn, tp = metrics.confusion_matrix(
        labels.astype(int),(preds > 0.5)*1
        ).ravel() # get the confusion matrix for 0.5 threshold
    accuracy_5  = (tp+tn)/(tn+fp+fn+tp)
    precision_5 = tp/(tp+fp) # also named purity
    recall_5    = tp/(tp+fn) # also named efficiency
    f1_5        = (2*precision_5*recall_5)/(precision_5+recall_5) 

    tn, fp, fn, tp = metrics.confusion_matrix(
        labels.astype(int),(preds > 0.7)*1
        ).ravel() # get the confusion matrix for 0.7 threshold
    accuracy_7  = (tp+tn)/(tn+fp+fn+tp)
    precision_7 = tp/(tp+fp) # also named purity
    recall_7    = tp/(tp+fn) # also named efficiency
    f1_7        = (2*precision_7*recall_7)/(precision_7+recall_7) 

    del labels
    del preds
    # Fill the batch_history dictionary with the obtained metrics
    history = {
        'accuracy_5': accuracy_5,
        'auc': auc,
        'precision_5': precision_5,
        'accuracy_3': accuracy_3,
        'precision_3': precision_3,
        'recall_3': recall_3,
        'f1_3': f1_3,
        'accuracy_5': accuracy_5,
        'precision_5': precision_5,
        'recall_5': recall_5,
        'f1_5': f1_5,
        'accuracy_7': accuracy_7,
        'precision_7': precision_7,
        'recall_7': recall_7,
        'f1_7': f1_7,
    }
    return history