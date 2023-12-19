import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os

class plotterABC:
    def __init__(self, runs, logs_dir, output_dir, mertics=["loss"], dataTypes=["training"],formates=["pdf"]):
        self.logs_dir = logs_dir
        self.output_dir = output_dir
        self.runs = runs
        self.formates = formates
        self.dataTypes = dataTypes
        self.mertics = mertics
        self.log_dfs = {}
        self.labels = {}

        for dataType in self.dataTypes:
            self.log_dfs[dataType] = {}
            self.labels[dataType] = {}
            for run in self.runs: 
                 self.labels[dataType][run] = self._extractLabel()
                 self.log_dfs[dataType][run] = pd.read_csv(f"{self.logs_dir}/run{run}/log_{dataType}.csv")

        for formate in self.formates:
            if not os.path.exists(f"{self.output_dir}/{formate}"):
                os.makedirs(f"{self.output_dir}/{formate}")

    def _extractLabel(self):
        label = "undefined"
        with open(self.config, 'r') as ymlfile:
            config = yaml.load(ymlfile, Loader=yaml.FullLoader)
            label = f"GNN - D={config['hid_dim']}, it={config['n_iters']}, lr={config['lr_c']}, bsize={config['batch_size']}"
            self.dataset = config['dataset']
        return label
        
    def makePlots(self):
        metric_dict = {}
        plt.clf()
        plt.figure()
        for metric in self.metrics:
            metric_dict[metric] = {}
            for dataType in self.dataTypes: 
                metric_dict[metric][dataType] = {}
                for run in self.runs: 
                    metric_dict[metric][dataType][run] = self.log_dfs[dataType][run][metric] if metric != 'auc' else  1- self.log_dfs[dataType][run][metric]
                    plt.plot(len(metric_dict[metric][dataType][run]), metric_dict[metric][dataType][run], label = self.labels[dataType][run])
        
            plt.title(f"data:{self.dataset} - GNN")
            plt.xlabel('epochs')
            if metric=='auc':
                ylabel = '1-AUC (lower the better)'
                plt.yscale('log')
                # plt.ylim([1e-6, 1.0])
            elif metric=='loss':
                ylabel = 'Loss (lower the better)'
                plt.yscale('log')
            else:
                ylabel=metric
            plt.ylabel(ylabel)
            #plt.ylim([0.2, 1.0])
            plt.legend()
            
            for formate in self.formates:
                if formate == "png":
                    plt.savefig(f"{self.output_dir}/{formate}/{metric}.{formate}", dpi=600)
                else:
                    plt.savefig(f"{self.output_dir}/{formate}/{metric}.{formate}")
            plt.close()
