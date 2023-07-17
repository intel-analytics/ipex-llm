import torch

def get_dummy_unet_additional_residuals():
    down_block_additional_residuals = []
    down_block_additional_residuals.extend([torch.zeros(2, 320, 64, 64)] * 3)
    down_block_additional_residuals.extend([torch.zeros(2, 320, 32, 32)])
    down_block_additional_residuals.extend([torch.zeros(2, 640, 32, 32)] * 2)
    down_block_additional_residuals.extend([torch.zeros(2, 640, 16, 16)])
    down_block_additional_residuals.extend([torch.zeros(2, 1280, 16, 16)] * 2)
    down_block_additional_residuals.extend([torch.zeros(2, 1280, 8, 8)] * 3)
    mid_block_additional_residual = torch.zeros(2, 1280, 8, 8)
    return down_block_additional_residuals, mid_block_additional_residual