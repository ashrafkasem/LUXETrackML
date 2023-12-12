import pandas as pd
import numpy as np
import os 

import argparse
import shutil
import glob

from core.data.track_segements import construct_segments
from core.policies.graph_policy import save_graph
from core.utiles import getFileList

input_dir = "/nfs/dust/ilc/user/amohamed/HTC/graph_building/LUXETrackML/done_analysis"
output_dir = "output_segments"

def get_cut_list(df):
    # Preselection parameters
    x_cut = 0.0526
    x_cut_epsilon = 0.0011
    y_cut = 0.001
    z_value = df['h_z'].min()
    return [x_cut, x_cut_epsilon, y_cut, z_value]


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Runs a NAF batch system for LUXE graph production', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', help='input directory', metavar='data_dir')
    parser.add_argument('--output_dir', help="output directory",  metavar='output_dir')
    # parser.add_argument('--start_event', help='the number of starting event', type=int, default=1)
    # parser.add_argument('--end_event', help='the number of ending event', type=int, default=1)
    parser.add_argument('--infile', help='if you dont run batch system you provide a single file', metavar='infile', default="")
    parser.add_argument('--batch', help="activate batch system submission",  action='store_true')
    parser.add_argument('--nevents', help='the number of events per batch job', type=int, default=-1)
    parser.add_argument('--splits', help='if you want to split the single events into smaller particle BXs', type=int, default=1)
    parser.add_argument('--bstra', help="if you want to bootstrap while sampling",  action='store_true')
    parser.add_argument('--evtnum', help='event number per batch job', type=int, default=-1)


    args = parser.parse_args()
    nevents_per_job = args.nevents

    if not args.batch:
        output_dir = args.output_dir
        if not os.path.exists(f"{output_dir}"):
            os.makedirs(output_dir)
        # read file
        events = pd.read_csv(args.infile)
        
        # give and event ID to each event
        events_list = np.unique(events['event'])
        
        if nevents_per_job > len(events_list):
            print(f"the file you provided has only {len(events_list)} events and you wanted to run over {nevents_per_job}")
            nevents_per_job = len(events_list)
        
        if args.evtnum >= 0 and nevents_per_job == 1:
            events = events[events['event']==args.evtnum]
        
        events_list = np.unique(events['event'])


        p_ids = events["particle_id"].unique()
        split_size = int(p_ids.shape[0]/args.splits)

        for snum, split in enumerate(range(args.splits)):
            # events = events[events['event'] == 2263]
            # p_ids = p_ids
            if args.bstra: 
                index = np.random.choice(p_ids.shape[0], split_size, replace=False)  
                selected_particles = p_ids[index]
            else: 
                print(f"making graph for the split {snum} which is choosing indexs from {snum*split_size} to {(snum+1) * split_size}")
                selected_particles = p_ids[snum*split_size:(snum+1) * split_size]
            print(f"number of particles to construct the tracks are {selected_particles.shape}")
            events = events[events["particle_id"].isin(selected_particles)]
            # print(selected_particles)
            # print(events.shape)

            for evt in events_list:
                # get stats and cut lists
                cut_list = get_cut_list(events[events["event"] == evt])
                # apply cuts/selections and obtain graphs
                segments, graph = construct_segments(events[events["event"] == evt], cut_list, getGraphandSegments=True)
            
                # save the graph file
                # print(graph.X)
                # print(graph.Ri.shape)
                # print(graph.Ro.shape)
                # print(graph.y.shape)
                save_graph(graph, f"{output_dir}/graph_{evt}_split_{snum}")
                
                # save the segments file
                segments.to_csv(f"{output_dir}/segments_{evt}_split_{snum}.csv")

                # Calculate statistics to later calculate the weights to be used in GNN
                nParticles = events[events["event"] == evt]['particle_id'].nunique()
                _,_,_,y = graph
                true_count = np.sum(y)
                fake_count = len(y) - np.sum(y)
                true_generated_count = segments['label'].sum()
                stats = [nParticles, true_count, fake_count, true_generated_count]
                np.save(f"{output_dir}/stats_{evt}_split_{snum}.npy", np.array(stats))
    
    else: 
        pass
        list_of_files = sorted(glob.glob(f"{args.data_dir}/truth_*.csv"))
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
        
        job_counter = 0
        for i, file_ in enumerate(list_of_files):
            start_event = int(file_.split("/")[-1].replace(".csv","").split("_")[1])
            end_event = int(file_.split("/")[-1].replace(".csv","").split("_")[2])
            temp_list_of_events= range(start_event,end_event+1)
            print(f"sumitting job for the following list of files:{range(start_event,end_event+1)}")
            for evt in temp_list_of_events:
                confDir = os.path.join(outdir,"job_"+str(evt))
                if not os.path.exists(confDir) :
                    os.makedirs(confDir)
                exec_ = open(confDir+"/exec.sh","w+")
                exec_.write("#"+"!"+"/bin/bash"+"\n")
                # exec_.write("eval "+'"'+"export PATH='"+path+":$PATH'"+'"'+"\n")
                # exec_.write("source "+anaconda+" hepML"+"\n")
                exec_.write(f"source {os.getcwd()}/venv/bin/activate"+"\n")

                exec_.write("cd "+confDir+"\n")

                exec_.write("echo 'running job' >> "+confDir+"/processing"+"\n")

                exec_.write("echo "+confDir+"\n")
                
                exec_.write(f"python {os.getcwd()}/runner_graph.py --data_dir ./ --output_dir {os.path.abspath(confDir)} --infile {os.path.abspath(file_)}  --nevents 1 --evtnum {evt} >> {confDir}/output.log")
                exec_.write("\n")
                
                # let the script deletes itself after finishing the job
                exec_.write(f"rm -rf {confDir}/processing"+"\n")
                exec_.write("echo 'done job' >> "+confDir+"/done"+"\n")              
                exec_.close()
                job_counter+=1

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
        subFile.write('Request_Memory = 8192M')
        subFile.write("\n")
        # subFile.write("+RequestRuntime = 6*60*60")
        subFile.write("\n")
        subFile.write("queue DIR matching dirs "+outdir+"/job_*/")
        subFile.close()
        submit_or_not = input(f"{job_counter+1} jobs created, do you want to submit? Please enter 'yes' or 'no':")
        if submit_or_not.lower().strip()[0] == "y":
            os.system("condor_submit "+subFilename)
        elif submit_or_not.lower().strip()[0] == "n":
            print(f" You decided not to submit from the script, you may go to {outdir}; excute #condor_submit {subFilename}")
