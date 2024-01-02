import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
from core.utiles import summary_statistics_across_dict_keys

class plotterABC:
    def __init__(
        self,
        runs,
        logs_dir,
        output_dir,
        metrics=["loss"],
        dataTypes=["training"],
        formates=["pdf"],
        logY={},
        Axis_limits={},
    ):
        self.logs_dir = logs_dir
        self.output_dir = output_dir
        self.runs = runs
        self.formates = formates
        self.dataTypes = dataTypes
        self.metrics = metrics
        self.log_dfs = {}
        self.labels = {}
        self.logY = logY
        self.Axis_limits = Axis_limits

        for dataType in self.dataTypes:
            self.log_dfs[dataType] = {}
            self.labels[dataType] = {}
            for run in self.runs:
                self.labels[dataType][run] = self._extractLabel(run)
                self.log_dfs[dataType][run] = pd.read_csv(
                    f"{self.logs_dir}/run{run}/log_{dataType}.csv"
                )

        for formate in self.formates:
            if not os.path.exists(f"{self.output_dir}/{formate}"):
                os.makedirs(f"{self.output_dir}/{formate}")

    def _extractLabel(self, run):
        label = "undefined"
        with open(f"{self.logs_dir}/run{run}/config.yaml", "r") as ymlfile:
            config = yaml.load(ymlfile, Loader=yaml.FullLoader)
            label = f"GNN - D={config['hid_dim']}, it={config['n_iters']}, lr={config['lr_c']}, bsize={config['batch_size']}"
            self.dataset = config["dataset"]
        return label

    def makePlots(self):
        metric_dict = {}
        for metric in self.metrics:
            plt.clf()
            plt.figure()
            metric_dict[metric] = {}
            for dataType in self.dataTypes:
                metric_dict[metric][dataType] = {}
                for run in self.runs:
                    metric_dict[metric][dataType][run] = (
                        self.log_dfs[dataType][run][metric]
                        if metric != "auc"
                        else 1 - self.log_dfs[dataType][run][metric]
                    )
                    plt.plot(
                        metric_dict[metric][dataType][run].index,
                        metric_dict[metric][dataType][run],
                        label=self.labels[dataType][run] + " " + dataType,
                    )

            plt.title(f"data:{self.dataset} - GNN")
            plt.xlabel("epochs")
            if metric == "auc":
                ylabel = "1-AUC (lower the better)"
            elif metric == "loss":
                ylabel = "Loss (lower the better)"
            else:
                ylabel = metric

            plt.ylabel(ylabel)

            if metric in self.logY and self.logY[metric] == True:
                plt.yscale("log")
            if metric in self.Axis_limits:
                plt.ylim(self.Axis_limits[metric])

            plt.legend()

            for formate in self.formates:
                if formate == "png":
                    plt.savefig(
                        f"{self.output_dir}/{formate}/{metric}.{formate}", dpi=600
                    )
                else:
                    plt.savefig(f"{self.output_dir}/{formate}/{metric}.{formate}")
            plt.close()

    def makeAvgPlots(self):
        metric_dict = {}
        for metric in self.metrics:
            plt.clf()
            plt.figure()
            metric_dict[metric] = {}
            for dataType in self.dataTypes:
                metric_dict[metric][dataType] = {}
                xidx = np.array([])
                for run in self.runs:
                    metric_dict[metric][dataType][run] = (
                        self.log_dfs[dataType][run][metric]
                        if metric != "auc"
                        else 1 - self.log_dfs[dataType][run][metric]
                    )
                    if xidx.shape[0] < len(metric_dict[metric][dataType][run].index): 
                        xidx = metric_dict[metric][dataType][run].index.to_numpy()

                summary = summary_statistics_across_dict_keys(metric_dict[metric][dataType])
                plt.plot(summary['mean'], label= f"{len(self.runs)} Avarage {dataType}")
                plt.fill_between(
                    xidx,
                    summary['min'],
                    summary['max'],
                    alpha=0.2
                )

            plt.title(f"data:{self.dataset} - GNN")
            plt.xlabel("epochs")
            if metric == "auc":
                ylabel = "1-AUC (lower the better)"
            elif metric == "loss":
                ylabel = "Loss (lower the better)"
            else:
                ylabel = metric

            plt.ylabel(ylabel)

            if metric in self.logY and self.logY[metric] == True:
                plt.yscale("log")
            if metric in self.Axis_limits:
                plt.ylim(self.Axis_limits[metric])

            plt.legend()

            for formate in self.formates:
                if formate == "png":
                    plt.savefig(
                        f"{self.output_dir}/{formate}/avg_{metric}.{formate}", dpi=600
                    )
                else:
                    plt.savefig(f"{self.output_dir}/avg_{formate}/{metric}.{formate}")
            plt.close()
