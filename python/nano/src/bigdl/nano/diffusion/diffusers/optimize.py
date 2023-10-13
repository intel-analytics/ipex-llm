#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import click

from huggingface_hub import snapshot_download
from diffusers import UNet2DConditionModel, AutoencoderKL

from ..utils.paths import *
from ..common.optimize import (
    save_unet_if_not_exist,
    save_vae_if_not_exist, save_controlnet_if_not_exist
)


def optimize(
        optimize_unet=True,
        optimize_vae=True,
        controlnet_path=None,
        model_path=None):
    """
    Trace a torch.nn.Module and convert it into an accelerated module for inference.
    For example, this function returns a PytorchOpenVINOModel when accelerator=='openvino'.
    :param low_memory: only valid when accelerator="jit"
        and ipex=True,model will use less memory during inference
    :cache_dir: the directory to save the converted model
    """
    if not os.path.exists(model_path):
        try:
            model_path = snapshot_download(model_path)
        except Exception:
            raise NoPathException(f"Could not find local `model_path` \
                in given local path, could not find the model on \
                Huggingface Hub by given `model_path` as repo_id either.")

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
    """The main function to transfer UNet to the nano UNet."""
    optimize(optimize_unet, optimize_vae, optimize_controlnet, model_path)


if __name__ == '__main__':
    main()
