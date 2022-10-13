from diffusers import StableDiffusionPipeline
import torch
import os

model_id = "CompVis/stable-diffusion-v1-4"


pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)

unet = pipe.unet

sample_latents = torch.randn((1, unet.in_channels, 64, 64),
		                             generator=None,
		                             device="cpu",
		                             dtype=torch.float32)
os.environ['DNNL_VERBOSE'] = '1'
unet_16 = unet.bfloat16()
a = unet_16(torch.cat([sample_latents] * 2), torch.tensor([980], dtype=torch.long),
            torch.randn((2, 77, 768), generator=None, device="cpu", dtype=torch.float32))
