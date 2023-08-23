import numpy as np
import pandas as pd

def cal_deta(hitpair):
    r1 = hitpair.r_1
    r2 = hitpair.r_2
    z1 = hitpair.z_1
    z2 = hitpair.z_2
    
    R1 = np.sqrt(r1**2 + z1**2)
    R2 = np.sqrt(r2**2 + z2**2)
    theta1 = np.arccos(z1/R1)
    theta2 = np.arccos(z2/R2)
#     theta1 = np.arctan(r1/z1)
#     theta2 = np.arctan(r2/z2)
    eta1 = -np.log(np.tan(theta1/2.0))
    eta2 = -np.log(np.tan(theta2/2.0))
    return eta1 - eta2

def calc_dphi(phi1, phi2):
    """Computes phi2-phi1 given in range [-pi,pi]"""
    dphi = phi2 - phi1
    dphi[dphi > np.pi] -= 2*np.pi
    dphi[dphi < -np.pi] += 2*np.pi
    return dphi


def create_segments(hits, gid_start, gid_end, gid_keys='layer_id'):
    segments = []
    hit_gid_groups = hits.groupby(gid_keys)

    # Loop over geometry ID pairs
    for gid1, gid2 in zip(gid_start, gid_end):
        hits1 = hit_gid_groups.get_group(gid1)
        hits2 = hit_gid_groups.get_group(gid2)

        # Join all hit pairs together
        hit_pairs = pd.merge(
            hits1.reset_index(), hits2.reset_index(),
            how='inner', on='evtid', suffixes=('_1', '_2'))
        
#         print(hit_pairs.columns)
        
        
        # Calculate coordinate differences
        dphi = calc_dphi(hit_pairs.phi_1, hit_pairs.phi_2)
        dz = hit_pairs.z_2 - hit_pairs.z_1
        dr = hit_pairs.r_2 - hit_pairs.r_1
        phi_slope = dphi / dr
        z0 = hit_pairs.z_1 - hit_pairs.r_1 * dz / dr
        deta = cal_deta(hit_pairs)

        # Identify the true pairs
        y = (hit_pairs.particle_id_1 == hit_pairs.particle_id_2) & (hit_pairs.particle_id_1 != 0)

        # Put the results in a new dataframe
        df_pairs = hit_pairs[['evtid', 'index_1', 'index_2', 'hit_id_1', 'hit_id_2', 'layer_id_1', 'layer_id_2',"particle_id_1",'particle_id_2', 'z_1', 'z_2',"r_1","r_2"]].assign(dphi=dphi, dz=dz, dr=dr, y=y, phi_slope=phi_slope, z0=z0, deta=deta)
        print('processed:', gid1, gid2, "True edges {} and Fake Edges {}".format(df_pairs[df_pairs['y']==True].shape[0], df_pairs[df_pairs['y']==False].shape[0]))
        segments.append(df_pairs)
    return segments