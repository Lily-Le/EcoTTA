import json
import os
import datetime
import torch
import numpy as np
import subprocess
# git config --global http.sslVerify false
# git config --global  --unset http.https://github.com.proxy
# git config --global  --unset https.https://github.com.proxy

def write_json(var_list,file_name):
    with open(file_name, 'w', encoding='UTF-8') as fp:
        try:
            fp.write(json.dumps(var_list, indent=2, ensure_ascii=False))
        except:
            x_dict = []
            for item in var_list:
                if isinstance(item, torch.Tensor):
                    item = item.cpu().detach().numpy().tolist()
                if isinstance(item,np.ndarray):
                    item=item.tolist()
                x_dict.append(item)
            fp.write(json.dumps(x_dict, indent=2, ensure_ascii=False))      

        

def writefile(filepath, filename):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    filewrite = open(os.path.join(filepath,filename) , 'a')
    return filewrite

# filetrainloss.write(str(epoch) + ' ' + str(loss) + '\n')

def filedel(filepath):
    for i in ['/testloss.txt','/testacc.txt','/trainloss.txt','/trainacc.txt','/testtime.txt','traintime.txt']:
        try:
            os.remove(filepath+i)
        except:
            pass

basename = "mylogfile"


def dir_rename(dir):
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    # filename = "_".join([basename, suffix]) # e.g. 'mylogfile_120508_171442'
    dst_dir=f'{dir}_last'
    if os.path.exists(dst_dir):
        os.rename(dst_dir,f'{dst_dir}_{suffix}')
    os.rename(dir,dst_dir)
    print(f'Last training result dir renamed to "{dir}_last"')
    
def mkd(args):
    try:
        os.mkdir('output')
    except:
        pass
    try:
        os.mkdir('output/' + args.codename.split('/')[0])
    except:
        pass
    try:
        # os.mkdir('output/' + args.codename.split('/')[0]+'/'+args.codename)
        os.makedirs('output/'+args.codename)
    except:
        pass
    
import sys
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass
 
# sys.stdout = Logger(a.log, sys.stdout)

#%%
import os
import pdb
def replaceDirName(rootDir):
    dirs = os.listdir(rootDir)
    dirs.sort()
    # 遍历每一个文件夹
    i=0
    for dir in dirs:
        # 输出老文件夹的名字
        print('oldname is:' + dir)  
                     
        # 主要目的是在数字统一为4位，不够的前面补0
        temp = "%04d" % int(i)   
        
        # 老文件夹的名字
        oldname = os.path.join(rootDir, dir) 
        # 新文件夹的名字     
        newname = os.path.join(rootDir, temp)     
        
        # 用新文件夹的名字替换老文件夹的名字
        # rename(*args, **kwargs):重命名文件或目录
        os.rename(oldname, newname)
        i=i+1

# %%
# rootDir='/home/cll/Workspace/data/cls/imagenette_rename/train'
# replaceDirName(rootDir)
# %%
def setup():
    # args.cuda = not args.cpu and torch.cuda.is_available()
    print("=== The available CUDA GPU will be used for computations.")
    memory_load = get_gpu_memory_usage()
    cuda_device = np.argmin(memory_load).item()
    torch.cuda.set_device(cuda_device)
    device = torch.cuda.current_device()
    # kwargs = {'num_workers': 2, 'pin_memory': True } if args.cuda else {}
    return device

def get_gpu_memory_usage():
    if sys.platform == "win32":
        curr_dir = os.getcwd()
        nvsmi_dir = r"C:\Program Files\NVIDIA Corporation\NVSMI"
        os.chdir(nvsmi_dir)
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used','--format=csv,nounits,noheader'])
        os.chdir(curr_dir)
    else:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used','--format=csv,nounits,noheader'])
    gpu_memory = [int(x) for x in result.decode('utf-8').strip().split('\n')]
    return gpu_memory