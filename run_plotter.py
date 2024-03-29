from core.plotting.plotter import plotterABC

if __name__ == "__main__":
    plotter = plotterABC(
        logs_dir="/Users/amohamed/dust/amohamed/HTC/graph_building/LUXETrackML/GNN_training_sampling_wl1_reg_adam_5perc/",
        output_dir="outputplotter_SGD",
        runs=[0],
        formates=["png"],
        dataTypes=["training","validation"],#["training"],#, "validation"],
        metrics=[
            "auc",
            "loss",
            "precision",
            "accuracy_3",
            "precision_3",
            "recall_3",
            "accuracy_5",
            "precision_5",
            "recall_5",
            "accuracy_7",
            "precision_7",
            "recall_7",
            "f1_7",
        ],
        # logY={"auc": True, "loss": True},
        # Axis_limits={"auc": [1e-6, 1], "loss": [0.2, 1]},
    )

    plotter.makePlots()
    # plotter.makeAvgPlots()
