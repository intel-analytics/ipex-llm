import os
from huggingface_hub import snapshot_download
from diffusers.utils import DIFFUSERS_CACHE
from stable_diffusion.utils import copy_model_from_snapshot, model_version_map
from diffusers import StableDiffusionPipeline

from stable_diffusion.pipelines import NanoStableDiffusionPipeline




def convert(pipeline_name, use_auth_token=None, local_model_path=None):
    if use_auth_token is None or len(use_auth_token) == 0:
        use_auth_token = None
    try:
        model_version, optimization_methods = pipeline_name.split("/")
        
        precision = "float16" if "FP16" in optimization_methods else "float32"
        device = "GPU" if "iGPU" in optimization_methods else "CPU"
        ipex = False
        low_memory = False
        
        if "OpenVINO" in optimization_methods:
            accelerator = "openvino"
        else:
            accelerator = "jit"
            if "IPEX" in optimization_methods:
                ipex = True
            if "Low-memory" in optimization_methods:
                low_memory = True

        # download snapshot and load from snapshot
        if local_model_path is None or local_model_path == "":
            model_id = model_version_map[model_version]["model_id"]
            cache_dir = snapshot_download(model_id, cache_dir="models", ignore_patterns=['*.ckpt', '*.safetensors'], token=use_auth_token)
        else:
            print(f"Trying to load local model...")
            cache_dir = local_model_path
        # load original diffusers model
        print(f"Loading model from {cache_dir}")
        pipe = StableDiffusionPipeline.from_pretrained(cache_dir)
        
        # create symlinks to cache files
        # os.makedirs(model_dir, exist_ok=True)
        # copy_model_from_snapshot(cache_dir, model_dir)

        # convert UNet model
        nano_pipe = NanoStableDiffusionPipeline(pipe)
        # Note that we have to set width, height sometimes if the model output is not 512 * 512
        nano_pipe.convert_pipeline(accelerator=accelerator, device=device, precision=precision, ipex=ipex, cache=True, low_memory=low_memory, cache_dir=cache_dir)
    except Exception as e:
        import traceback
        return traceback.format_exc()
    return "Model optimization finished."


   
    
    

