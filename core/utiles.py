import os 

def getFileList(path,start_event,end_event, pattern = "e0gpc_7.0_%04d_positrons_edm4hep.tar.gz"):
    FileList = []

    #return nothing if path is a file
    if os.path.isfile(path):
        return []

    FileList = sorted([os.path.join(path,pattern) % i for i in range(start_event,end_event+1)])

    return FileList

