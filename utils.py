import torch
import numpy as np
import random
import pathlib
import datetime

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def make_dirs(prefix):
    # Make a folder to save all output
    subFolderName = '%s'%(datetime.datetime.now().strftime("%y%m%d%H%M%S")) 
    if prefix == '':
        FolderName = '%s/%s/' % (1, subFolderName)
    else:
        FolderName = '%s/%s/' % ( prefix, subFolderName)
    model_dir = '%s/model/%s/' % ('results', FolderName)
    log_dir = '%s/logs/%s/' % ('results', FolderName)
    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True) 
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True) 
    
    return FolderName, model_dir, log_dir