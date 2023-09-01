import numpy as np
import torch
import torch.distributed as dist

@torch.no_grad()
def gather_same_shape_tensors(tensor):
    """Performs all_gather operation on the provided tensors."""
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

@torch.no_grad()
def gather_diff_shape_tensors(tensor, shape_list):
    """Performs all_gather operation on the provided tensors."""
    tensors_gather = []
    for i in range(len(shape_list)):
        tensors_gather.append(torch.ones(shape_list[i]).to(tensor.device))
    print("tensors gather:", tensors_gather)
    dist.all_gather(tensors_gather, tensor, async_op=False)
    return tensors_gather