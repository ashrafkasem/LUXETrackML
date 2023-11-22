import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from track_parameters import xz_angle, yz_angle, doublet_criteria_check


class LUXETrackSegmentor:
    def __init__(self, data_dir, output_dir, start_event, end_event):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.start_event = start_event
        self.end_event = end_event
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def getFileList(self, pattern="e0gpc_7.0_%04d_positrons_edm4hep.tar.gz"):
        FileList = []

        # return nothing if path is a file
        if os.path.isfile(self.data_dir):
            return []

        FileList = sorted(
            [
                os.path.join(self.data_dir, pattern) % i
                for i in range(self.start_event, self.end_event + 1)
            ]
        )

        return FileList

    def slice_percentage(self, df, percentage=0.05):
        selected_particles = df["particle_id"].sample(frac=0.05).values
        df = df[df["particle_id"].isin(selected_particles)]
        return df

    def get_segments(
        self,
        hits,
        gid_keys,
        gid_start,
        gid_end,
        dy_threshold=0.1,
        yz_angel_threshold=0.025,
        xz_angle_upper=0.3,
        xz_angle_lower=0,
        chunk_size=1000,
    ):
        z_value = hits["h_z"].min()
        # get the y-slice of the detector
        # Group hits by geometry ID
        hit_gid_groups = hits.groupby(gid_keys)
        outfiles_list = []
        summary_stats = {}
        # Loop over geometry ID pairs
        for gid1, gid2 in zip(gid_start, gid_end):
            print(gid1, gid2)
            summary_stats[(gid1, gid2)] = {}
            summary_stats[(gid1, gid2)]["before"] = [0, 0]
            summary_stats[(gid1, gid2)]["after"] = [0, 0]
            hits1 = hit_gid_groups.get_group(gid1)
            hits2 = hit_gid_groups.get_group(gid2)
            if chunk_size == -1:
                chunks_hits1 = [hits1]
            else:
                chunks_hits1 = [
                    hits1[i : i + chunk_size] for i in range(0, len(hits1), chunk_size)
                ]
            for nchunk, chunk_hits1 in enumerate(chunks_hits1):
                #             if nchunk > 0 : break
                print("chunk_hits1", chunk_hits1.shape)
                print("chunk number:", nchunk, "/", len(chunks_hits1))
                # Join all hit pairs together
                hit_pairs = pd.merge(
                    chunk_hits1.reset_index(),
                    hits2.reset_index(),
                    how="inner",
                    on="event",
                    suffixes=("_1", "_2"),
                )

                # Calculate coordinate differences

                hit_pairs["xz_angle"] = xz_angle(hit_pairs)
                hit_pairs["yz_angle"] = yz_angle(hit_pairs)
                hit_pairs["dy"] = np.abs(hit_pairs["h_y_1"] - hit_pairs["h_y_2"])
                hit_pairs["y"] = (
                    hit_pairs.particle_id_1 == hit_pairs.particle_id_2
                ) & (hit_pairs.particle_id_1 != 0)
                summary_stats[(gid1, gid2)]["before"][0] += (
                    hit_pairs["y"] == True
                ).sum()
                summary_stats[(gid1, gid2)]["before"][1] += (
                    hit_pairs["y"] == False
                ).sum()

                hit_pairs = hit_pairs[
                    (hit_pairs["xz_angle"] < xz_angle_upper)
                    & (hit_pairs["xz_angle"] > xz_angle_lower)
                    & (np.abs(hit_pairs["yz_angle"]) < yz_angel_threshold)
                    & (hit_pairs["dy"] < dy_threshold)
                ]
                hit_pairs = doublet_criteria_check(hit_pairs, z_value)
                hit_pairs = hit_pairs[
                    (hit_pairs["dy"] / hit_pairs["x_at_z_value"] < 0.001)
                    & (hit_pairs["dx/x0"] < 0.1)
                ]
                summary_stats[(gid1, gid2)]["after"][0] += (
                    hit_pairs["y"] == True
                ).sum()
                summary_stats[(gid1, gid2)]["after"][1] += (
                    hit_pairs["y"] == False
                ).sum()
                # Identify the true pairs
                # Put the results in a new dataframe
                hit_pairs[
                    [
                        "event",
                        "index_1",
                        "index_2",
                        "hit_id_1",
                        "hit_id_2",
                        "h_layer_1",
                        "h_layer_2",
                        "dx/x0",
                        "satisfied",
                        "dx",
                        "x_at_z_value",
                        "y_at_z_value",
                        "xz_angle",
                        "yz_angle",
                        "dy",
                        "y",
                    ]
                ].to_csv(
                    f"{self.output_dir}/{gid1}_{gid2}_chunk_{nchunk}.csv", index=False
                )
                print(hit_pairs.shape)
                outfiles_list.append(
                    f"{self.output_dir}/{gid1}_{gid2}_chunk_{nchunk}.csv"
                )
                #             plt.show()
                del hit_pairs
            del hits1, hits2
        with open(f"{self.output_dir}/summary_stat.pkl", "wb") as fp:
            pickle.dump(summary_stats, fp)
            print("summary_stats dictionary saved successfully to file")
        #             segments[-1].plot.hist(column=["dy"], by="y", figsize=(10, 8))
        return outfiles_list


if __name__ == "__main__":
    indir = "/home/amohamed/ML/LUXETrackML/output_luxe_data/"
    # Define the paths to the files that contain the hits and particles information
    # hitfiles = getFileList(indir, 1, 1, "hits_%04d.csv")
    # particlefiles = getFileList(indir, 1, 1, "particles_%04d.csv")
    segmentor = LUXETrackSegmentor(indir,
                                   "./segments",
                                   0,
                                   1)

    truthfiles = segmentor.getFileList(indir, 1, 1, "truth_%04d.csv")
    hits = pd.concat(map(pd.read_csv, truthfiles), ignore_index=True)
    # particles_df = pd.concat(map(pd.read_csv, particlefiles), ignore_index=True)
    # truth_df = pd.concat(map(pd.read_csv, truthfiles), ignore_index=True)

    gid_keys = "h_layer"
    n_det_layers = 4
    gid_start = np.arange(1, n_det_layers)
    gid_end = np.arange(2, n_det_layers + 1)


    outfiles_list = segmentor.get_segments(hits,
                        gid_keys, gid_start, gid_end,
                        dy_threshold= 0.1 ,
                        yz_angel_threshold=0.025,
                        xz_angle_upper=0.3, xz_angle_lower=0,
                        chunk_size = 400)
