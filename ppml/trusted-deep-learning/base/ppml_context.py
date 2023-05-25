# +
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch import nn
import os

# get distributed conf
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))

class PPMLContext:
    def __init__(self, k8s_enabled = True):
        if k8s_enabled:
            print("Using distributed PyTorch with {} backend".format(
                "GLOO"), flush=True)
            dist.init_process_group(backend=dist.Backend.GLOO)
    def get_distributed(self, model, train_dataloader, valid_dataloader = None):
        #get distributed model
        Distributor = nn.parallel.DistributedDataParallel
        model1 = Distributor(model, find_unused_parameters=True)
        
        #get distributed data_loader
        train_dataloader_kwargs = train_dataloader.__dict__.copy()  # 获取train_dataloader1的当前配置
        train_sampler = DistributedSampler(
            train_dataloader_kwargs['dataset'], num_replicas=WORLD_SIZE, rank=RANK, shuffle=True, drop_last=False)
        train_dataloader1 = DataLoader(
            train_dataloader_kwargs['dataset'],
            batch_size=train_dataloader_kwargs['batch_size'],
            collate_fn=train_dataloader_kwargs['collate_fn'],
            sampler=train_sampler,
        )
        
        if valid_dataloader != None:
            valid_dataloader_kwargs = valid_dataloader.__dict__.copy()  # 获取train_dataloader1的当前配置
            valid_sampler = DistributedSampler(
                valid_dataloader_kwargs['dataset'], num_replicas=WORLD_SIZE, rank=RANK, shuffle=True, drop_last=False)
            valid_dataloader1 = DataLoader(
                valid_dataloader_kwargs['dataset'],
                batch_size=valid_dataloader_kwargs['batch_size'],
                collate_fn=valid_dataloader_kwargs['collate_fn'],
                sampler=valid_sampler,
            )
        return model1, train_dataloader1, valid_dataloader1
# -




