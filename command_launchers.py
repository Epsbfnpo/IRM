import subprocess
import time
import torch
import os

def local_launcher(commands):
    for cmd in commands:
        subprocess.call(cmd, shell=True)

def dummy_launcher(commands):
    for cmd in commands:
        print(f'Dummy launcher: {cmd}')

def multi_gpu_launcher(commands):
    print('WARNING: using experimental multi_gpu_launcher.')
    try:
        available_gpus = [x for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if x != '']
    except Exception:
        available_gpus = [str(x) for x in range(torch.cuda.device_count())]
    n_gpus = len(available_gpus)
    procs_by_gpu = [None]*n_gpus
    while len(commands) > 0:
        for idx, gpu_idx in enumerate(available_gpus):
            proc = procs_by_gpu[idx]
            if (proc is None) or (proc.poll() is not None):
                cmd = commands.pop(0)
                new_proc = subprocess.Popen(f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}', shell=True)
                procs_by_gpu[idx] = new_proc
                break
        time.sleep(1)
    for p in procs_by_gpu:
        if p is not None:
            p.wait()

REGISTRY = {'local': local_launcher, 'dummy': dummy_launcher, 'multi_gpu': multi_gpu_launcher}
try:
    from domainbed import facebook
    facebook.register_command_launchers(REGISTRY)
except ImportError:
    pass
