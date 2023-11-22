import numpy as np
from core.data.luxe_connector import LUXEDataConnector

if __name__ == "__main__":
    LUXEData = LUXEDataConnector(
        data_dir = "/home/amohamed/dust/amohamed/HTC/dataframes_new",
        output_dir = "output_luxe_data",
        start_event= 1,
        end_event= 5
        
    )
    
    truth = LUXEData.readfiles()

    particles_vars =  ['event', 'particle_id', 'p_Energy',
                       'p_Charge', 'p_Time', 'p_Mass', 'p_vx',
                        'p_vy', 'p_vz', 'p_end_vx','p_end_vy',
                        'p_end_vz', 'p_px', 'p_py', 'p_pz',
                        'p_end_px', 'p_end_py','p_end_pz', 
                        'p_isOverlay', 'p_isStopped','p_isCreatedInSimulation',
                        'p_isBackscatter','p_vertexIsNotEndpointOfParent',
                        'p_isDecayedInTracker','p_isDecayedInCalorimeter', 'p_hasLeftDetector','nhits']
    
    
    hits_vars = ['event', 'hit_id', 'h_CellID', 'h_EDep', 'h_Time', 'h_PathLength',
       'h_Quality', 'h_isOverlay', 'h_x', 'h_y', 'h_z', 'h_px', 'h_py', 'h_pz','h_layer', 'h_stave', "isProducedBySecondary"]

    truth_vars = ['event','hit_id', 'particle_id','h_x', 'h_y', 'h_z','h_px', 'h_py', 'h_pz','h_layer', 'h_stave', 'p_px', 'p_py', 'p_pz', "isProducedBySecondary"]

    particles = LUXEData.make_partial_df(truth,particles_vars)
    hits = LUXEData.make_partial_df(truth,hits_vars)
    truth = LUXEData.make_partial_df(truth, truth_vars)

    # add transverse momentum: pt into the dataframe
    # px = particles.p_px
    # py = particles.p_px
    # pt = np.sqrt(px**2 + py**2)
    # particles = particles.assign(p_pt=pt)
    
    # x = hits.h_x
    # y = hits.h_y
    # z = hits.h_z
    # rho = np.sqrt(x**2 + y**2)
    # r = np.sqrt(x**2 + y**2 + z**2)
    # phi = np.arctan2(rho, z)
    # theta = np.arctan2(y, x) 
    # hits = hits.assign(r=r, phi=phi, rho=rho, theta = theta)
    LUXEData.write_outfiles(hits, particles, truth)
    1==1