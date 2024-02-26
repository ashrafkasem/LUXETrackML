import matplotlib
matplotlib.use('agg')  # Use the 'agg' backend
import os
# Turn off warnings and errors due to TF libraries
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
import tensorflow as tf
from core.tools import *
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class TrainingError(Exception):
    """Training error class."""
    pass


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
    
    os.makedirs(config["output_dir"], exist_ok=True)

    # setup model
    model = GNN()
    model_dir = os.path.join(f"{config['run_dir']}/run{config['run_number']}", "tfmodel")
    try: 
        loaded_model = tf.keras.models.load_model(model_dir)
        for idx, params in enumerate(loaded_model.trainable_variables):
            model.trainable_variables[idx].assign(params)
        print(
            str(datetime.datetime.now()), f"Model is loaded from {model_dir}"
            )
    except TrainingError: 
        loaded_model, check_point = load_params(model, f"{config['run_dir']}/run{config['run_number']}")
        for idx, params in enumerate(loaded_model.trainable_variables):
            model.trainable_variables[idx].assign(params)
        print(
            str(datetime.datetime.now()), f"Model is loaded from the last checkpoint {check_point}"
            )
    except Exception as e:
        print(
            str(datetime.datetime.now()), f"could not load model from {model_dir}"
            )
        raise e

    datasets = get_dataset(config['inference_dir'], config['n_inference'])

    # Initialize lists to store true labels and predictions
    all_y_true = []
    all_y_pred = []

    # Iterate over each dataset in test_data
    for dataset in datasets:
        X, Ri, Ro, y = dataset
        # Get predictions from the model
        y_ = model([X, Ri, Ro])
        pred = y_.numpy()
        # Convert predictions to binary (0 or 1)
        pred[pred < 0.5] = 0
        pred[pred >= 0.5] = 1
        pred = pred.flatten()
        
        # Append true labels and predictions to the lists
        all_y_true.extend(y)
        all_y_pred.extend(pred)
    
    # Convert lists to NumPy arrays
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    # Compute evaluation metrics
    accuracy = accuracy_score(all_y_true, all_y_pred)
    precision = precision_score(all_y_true, all_y_pred)
    recall = recall_score(all_y_true, all_y_pred)
    f1 = f1_score(all_y_true, all_y_pred)
    
    # Compute confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    
    # Open a file in write mode
    with open(f"{config['output_dir']}/evaluation_metrics.txt", "w") as file:
        # Write evaluation metrics into the file using f-strings
        file.write(f"Accuracy: {accuracy}\n")
        file.write(f"Precision: {precision}\n")
        file.write(f"Recall: {recall}\n")
        file.write(f"F1 Score: {f1}\n")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', )
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    plt.savefig(f"{config['output_dir']}/CM.png")
    # plt.show()

    # Plot normalized confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, cmap='Blues', fmt='.2%',xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig(f"{config['output_dir']}/normalizedCM.png")
    # plt.show()
