import click
from huggingface_hub import snapshot_download
from diffusers import UNet2DConditionModel, AutoencoderKL

from ..common.optimize import save_unet_if_not_exist, save_vae_if_not_exist, save_controlnet_if_not_exist
from ..utils.paths import *


def optimize(
        optimize_unet=True, 
        optimize_vae=True, 
        controlnet_path=None,
        model_path=None):
    '''
    Optimizes a diffusers format pipeline given a repo id or local file path
        Parameters:
            optimize_unet: if True, will optimize UNet
            optimize_vae: if True, will optimize VAE
            controlnet_path: if not None, will optimize ControlNet in this local path
            model_path: the local path or the repo id of the model
    '''
    if not os.path.exists(model_path):
        try:
            model_path = snapshot_download(model_path)
        except Exception:
            raise NoPathException(f"Could not find local `model_path` in given local path, could not find the model on Huggingface Hub by given `model_path` as repo_id either.")

    if optimize_unet:  
        unet_local_path = os.path.join(model_path, "unet")
        unet = UNet2DConditionModel.from_pretrained(unet_local_path)
        save_unet_if_not_exist(unet)
    if optimize_vae:
        vae_local_path = os.path.join(model_path, "vae")        
        vae = AutoencoderKL.from_pretrained(vae_local_path)
        save_vae_if_not_exist(vae)

    if controlnet_path is not None:
        save_controlnet_if_not_exist(controlnet_path)
        
@click.command()
@click.option('--optimize_unet', default=True, is_flag=True)
@click.option('--optimize_vae', default=True, is_flag=True)
@click.option('--optimize_controlnet', type=str)
@click.option('-m', '--model_path', default=None, type=str)
def main(optimize_unet, optimize_vae, optimize_controlnet, model_path):
    optimize(optimize_unet, optimize_vae, optimize_controlnet, model_path)


if __name__ == '__main__':
    main()
