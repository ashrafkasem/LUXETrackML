import os
import argparse
import shutil
import glob
from core.data.luxe_connector import LUXEDataConnector
from core.utiles import getFileList


if __name__ == "__main__":
    # Define a custom argument type for a list of strings
    def list_of_strings(arg):
        return arg.split(',')
    parser = argparse.ArgumentParser(description='Runs a NAF batch system for LUXE data analysis', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', help='input directory', metavar='data_dir')
    parser.add_argument('--output_dir', help="output directory",  metavar='output_dir')
    parser.add_argument('--Files', help='Path of a file',type=list_of_strings)
    # parser.add_argument('--start_event', help='the number of starting event', type=int, default=1)
    # parser.add_argument('--end_event', help='the number of ending event', type=int, default=1)
    parser.add_argument('--batch', help="activate batch system submission",  action='store_true')
    parser.add_argument('--nfiles', help='the number of files per batch job', type=int, default=1)

    args = parser.parse_args()


    if not args.batch:
            
        LUXEData = LUXEDataConnector(
            data_dir = args.data_dir,
            output_dir = args.output_dir,
            # start_event= args.start_event,
            # end_event= args.end_event
            
        )
        print(args.Files)
        truth = LUXEData.readfiles(args.Files)
        print(truth)

        # particles_vars =  ['event', 'particle_id', 'p_Energy',
        #                    'p_Charge', 'p_Time', 'p_Mass', 'p_vx',
        #                     'p_vy', 'p_vz', 'p_end_vx','p_end_vy',
        #                     'p_end_vz', 'p_px', 'p_py', 'p_pz',
        #                     'p_end_px', 'p_end_py','p_end_pz', 
        #                     'p_isOverlay', 'p_isStopped','p_isCreatedInSimulation',
        #                     'p_isBackscatter','p_vertexIsNotEndpointOfParent',
        #                     'p_isDecayedInTracker','p_isDecayedInCalorimeter', 'p_hasLeftDetector','nhits']
        
        
        # hits_vars = ['event', 'hit_id', 'h_CellID', 'h_EDep', 'h_Time', 'h_PathLength',
        #    'h_Quality', 'h_isOverlay', 'h_x', 'h_y', 'h_z', 'h_px', 'h_py', 'h_pz','h_layer', 'h_stave', "isProducedBySecondary"]

        truth_vars = ['event','hit_id', 'particle_id','h_x', 'h_y', 'h_z','h_px', 'h_py', 'h_pz','h_layer', 'h_stave', 'p_px', 'p_py', 'p_pz', "isProducedBySecondary"]

        # particles = LUXEData.make_partial_df(truth,particles_vars)
        # hits = LUXEData.make_partial_df(truth,hits_vars)
        print("writting the output")
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
        LUXEData.write_outfiles(truth, type="truth")

    else: 
        list_of_files = sorted(glob.glob(f"{args.data_dir}/*_positrons_edm4hep.tar.gz"))
        outdir = os.path.abspath(args.output_dir)
        logs = os.path.abspath(f"{outdir}/logs")

        if os.path.exists(f"{outdir}"):
            answer = input(f"{outdir} already exists, do you want to delete it? Please enter 'yes' or 'no':")
            if answer.lower().strip()[0] == "y":
                print(f"removing the old output dir {outdir}")
                shutil.rmtree(outdir)
            elif answer.lower().strip()[0] == "n":
                print("You decided to keep the old out dir ")

        if not os.path.exists(f"{outdir}"):
            os.makedirs(f"{outdir}")

        if not os.path.exists(f"{logs}"):
            os.makedirs(f"{logs}")
        temp_list_of_files = []
        for i, file_ in enumerate(list_of_files):
            temp_list_of_files.append(file_)
            if ((i !=0) and ((i+1) % args.nfiles == 0)) or (i == len(list_of_files)-1):
                print(f"sumitting job for the following list of files:{temp_list_of_files}")

                confDir = os.path.join(outdir,"job_"+str(i+1))
                if not os.path.exists(confDir) : 
                    os.makedirs(confDir)
                

                exec_ = open(confDir+"/exec.sh","w+")
                exec_.write("#"+"!"+"/bin/bash"+"\n")
                # exec_.write("eval "+'"'+"export PATH='"+path+":$PATH'"+'"'+"\n")
                # exec_.write("source "+anaconda+" hepML"+"\n")
                exec_.write("source /nfs/dust/ilc/user/amohamed/HTC/graph_building/LUXETrackML/venv/bin/activate"+"\n")

                exec_.write("cd "+confDir+"\n")

                exec_.write("echo 'running job' >> "+confDir+"/processing"+"\n")

                exec_.write("echo "+confDir+"\n")
                files = ",".join(str(os.path.abspath(e)) for e in temp_list_of_files)
                exec_.write(f"python {os.getcwd()}/runner.py --Files {files} --output_dir {confDir}")
                exec_.write("\n")
                
                # let the script deletes itself after finishing the job
                exec_.write(f"rm -rf {confDir}/processing"+"\n")
                exec_.write("echo 'done job' >> "+confDir+"/done"+"\n")              
                exec_.close()
                temp_list_of_files = []

            else: 
                continue

        subFilename = os.path.join(outdir,"submitAllJobs.conf")
        subFile = open(subFilename,"w+")
        subFile.write("executable = $(DIR)/exec.sh"+"\n")
        subFile.write("universe =  vanilla")
        subFile.write("\n")
        subFile.write("should_transfer_files = YES")
        subFile.write("\n")
        subFile.write("log = "+"{}/job_$(Cluster)_$(Process).log".format(os.path.abspath(logs)))
        subFile.write("\n")
        subFile.write("output = "+"{}/job_$(Cluster)_$(Process).out".format(os.path.abspath(logs)))
        subFile.write("\n")
        subFile.write("error = "+"{}/job_$(Cluster)_$(Process).err".format(os.path.abspath(logs)))
        subFile.write("\n")
        subFile.write("when_to_transfer_output   = ON_EXIT")
        subFile.write("\n")
        subFile.write('Requirements  = ( OpSysAndVer == "CentOS7")')
        subFile.write("\n")
        # subFile.write("+RequestRuntime = 6*60*60")
        subFile.write("\n")
        subFile.write("queue DIR matching dirs "+outdir+"/job_*/")
        subFile.close()
        submit_or_not = input(f"{i+1} jobs created, do you want to submit? Please enter 'yes' or 'no':")
        if submit_or_not.lower().strip()[0] == "y":
            os.system("condor_submit "+subFilename)
        elif submit_or_not.lower().strip()[0] == "n":
            print(f" You decided not to submit from the script, you may go to {outdir}; excute #condor_submit {subFilename}")