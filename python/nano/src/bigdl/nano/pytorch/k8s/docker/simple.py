import torch.distributed as dist
dist.init_process_group("gloo")
dist.barrier()
print(f"rank is {dist.get_rank()}, world_size is {dist.get_world_size()}")
dist.destroy_process_group()
