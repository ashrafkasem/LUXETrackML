import pandas as pd
import numpy as np
from core.data.data_connector import DataConnector 
from core.utiles import getFileList
import os

class LUXEDataConnector(DataConnector): 
    def __init__(self, data_dir, output_dir):#, start_event, end_event):
        self.data_dir = data_dir
        self.output_dir = output_dir
        # self.start_event = start_event
        # self.end_event = end_event
        self.layer_map = {1:[0,4000], 2:[4000,4100], 3:[4100,4200], 4:[4200,5000]}
        self.stave_map = {1:[3955,3961], 2:[3961,4055], 3:[4055,4061], 4:[4061,4155], 5: [4155,4161], 6: [4161,4255], 7:[4255,4261 ], 8:[4261,4300]}

    def readfiles(self, FileList):
        # FileList = getFileList(self.data_dir, self.start_event, self.end_event)

        dfList = []

        for File in FileList:
            temp_df = self.readfile(File)
            temp_df = self.filter_data(temp_df)

            dfList.append(temp_df)

        total_df = pd.concat(dfList,  axis=0, ignore_index=True)

        self.set_detector_maps(total_df, "h_z")

        total_df["h_layer"] = self.make_layer()
        total_df["h_stave"] = self.make_stave()
        total_df['nhits'] = self.make_nhits(total_df, groupvar =["event","particle_id"], count_var= "particle_id")

        return total_df

    def readfile(self, File):
        return pd.read_pickle(File)

    def make_nhits(self, df, groupvar = ["event","particle_id"],  count_var= "particle_id"):
        return df.groupby(groupvar)[count_var].transform('count')
    
    def make_partial_df(self,truth_df, hits_vars):
        df = truth_df[hits_vars].drop_duplicates().reset_index(drop=True)
        return df.convert_dtypes()
       
    def set_detector_maps(self, df, col):
        self.layer_conditions = []
        self.layer_choices = []
        self.stave_conditions = []
        self.stave_choices = []
        for key, val in self.layer_map.items():
            # detect the boundaries
            if len(val) != 2 : raise Exception("Sorry, the layer map should have upper and lower bound for each layer") 
            cond = ((df[col] >= val[0])&( df[col]< val[1]))
            self.layer_conditions.append(cond)
            self.layer_choices.append(key)       

        for key, val in self.stave_map.items():
            # detect the boundaries
            if len(val) != 2 : raise Exception("Sorry, the layer map should have upper and lower bound for each layer") 
            cond = ((df[col] >= val[0])&( df[col]< val[1]))
            self.stave_conditions.append(cond)
            self.stave_choices.append(key)       

    def make_layer(self):
        return np.select(self.layer_conditions, self.layer_choices, default=np.nan)
        
    def make_stave(self):
        return np.select(self.stave_conditions, self.stave_choices, default=np.nan)

    def make_truth():
        pass

    def filter_data(self, df):
        df = df[df['h_CellID'].notna()]
        return df

    def write_outfiles(self, df, type="truth"):#, particles_df, truth_df):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        low_evt = df["event"].min()
        high_evt = df["event"].max()
        df.to_csv(os.path.join(self.output_dir, f"{type}_%04d_%04d.csv" %(low_evt,high_evt) ), index=False)
        # particles_df.to_csv(os.path.join(self.output_dir, "particles_%04d_%04d.csv" %(low_evt,high_evt) ), index=False)
        # truth_df.to_csv(os.path.join(self.output_dir, "truth_%04d_%04d.csv" %(low_evt,high_evt) ), index=False)
    
