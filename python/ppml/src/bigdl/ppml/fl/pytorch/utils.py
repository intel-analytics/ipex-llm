import torch
from torch import nn

def set_one_like_parameter(model: nn.Module):
    for param in model.parameters():
        param.data = nn.parameter.Parameter(torch.ones_like(param))