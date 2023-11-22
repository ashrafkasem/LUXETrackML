import pandas as pd
import numpy as np

def doublet_criteria_check(hits_df, z_value):
#     print(hits_df.columns)
    hits_df = get_coordinate_at_z_value(hits_df,z_value)
    hits_df["dx/x0"] = (hits_df.h_x_2 - hits_df.h_x_1) /hits_df['x_at_z_value']
    
    mean = np.mean(hits_df["dx/x0"]) 
    std = np.std(hits_df["dx/x0"])
    print(mean, std)
    hits_df["satisfied"] = False
    
    hits_df["dx"] = hits_df.h_x_2 - hits_df.h_x_1

    satisfied_mask = abs(hits_df["dx/x0"] - mean ) <= 3*std
    hits_df.loc[satisfied_mask, "satisfied"] = True
    return hits_df

def get_coordinate_at_z_value(hits_df, z_value):

    dx_p1_p2 = hits_df.h_x_2 - hits_df.h_x_1
    dy_p1_p2 = hits_df.h_y_2 - hits_df.h_y_1
    dz_p1_p2 = hits_df.h_z_2 - hits_df.h_z_1
    
    x_at_z_value = hits_df.h_x_2 - dx_p1_p2 * abs(hits_df.h_z_2 - z_value) / dz_p1_p2
    y_at_z_value = hits_df.h_y_2 - dy_p1_p2 * abs(hits_df.h_z_1 - z_value) / dz_p1_p2
    
    hits_df['x_at_z_value'] = x_at_z_value
    hits_df['y_at_z_value'] = y_at_z_value    
    
    z1_mask = (hits_df.h_z_1 == z_value)
    z2_mask = (hits_df.h_z_2 == z_value)
        
    hits_df.loc[z1_mask, 'x_at_z_value'] = hits_df.loc[z1_mask,"h_x_1"]
    hits_df.loc[z1_mask, 'y_at_z_value'] = hits_df.loc[z1_mask, "h_y_1"]
    hits_df.loc[z2_mask, 'x_at_z_value'] = hits_df.loc[z2_mask, "h_x_2"]
    hits_df.loc[z2_mask, 'y_at_z_value'] = hits_df.loc[z2_mask,"h_y_2"]
    return hits_df

def xz_angle(hits):
    """Returns the angle in the xz-plane with respect to beam axis in z direction.
    :return
        angle in the xz plane.
    """
    return np.arctan2((hits.h_x_2 - hits.h_x_1),
                    (hits.h_z_2 - hits.h_z_1))

def yz_angle(hits):
    """Returns the angle in the xz-plane with respect to beam axis in z direction.
    :return
        angle in the yz plane.
    """
    return np.arctan2((hits.h_y_2 - hits.h_y_1),
                (hits.h_z_2 - hits.h_z_1))


def optimize_datatype(df: pd.DataFrame) -> pd.DataFrame:
    floats = df.select_dtypes(include=['float64']).columns.tolist()
    df[floats] = df[floats].apply(pd.to_numeric, downcast='float')
    ints = df.select_dtypes(include=['int64']).columns.tolist()
    df[ints] = df[ints].apply(pd.to_numeric, downcast='integer')
    return df


