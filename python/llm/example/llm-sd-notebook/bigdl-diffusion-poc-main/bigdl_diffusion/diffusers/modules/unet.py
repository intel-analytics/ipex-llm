from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from diffusers.models.unet_2d_condition import UNet2DConditionOutput

from bigdl.nano.pytorch.model import AcceleratedLightningModule
from bigdl.nano.deps.openvino.pytorch.model import PytorchOpenVINOModel


from bigdl_diffusion.utils.pipeline import get_dummy_unet_additional_residuals


class NanoUNet(torch.nn.Module):
    """
    an UNet wrapper to implement additional operations in `forward()` method

    ...

    Attributes
    ----------
    model : AcceleratedLightningModule
        the original nano-optimized model
    """
    def __init__(self, model: AcceleratedLightningModule):
        super().__init__()
        self.model = model
        self.config = model.config
        self.in_channels = model.in_channels


    def forward(
            self, 
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            class_labels: Optional[torch.Tensor] = None,
            timestep_cond: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
            mid_block_additional_residual: Optional[torch.Tensor] = None,
            return_dict: bool = True,
    ):
        extra_args = []
        if cross_attention_kwargs is not None:
            attn_scale = cross_attention_kwargs.pop("scale", None)
            if attn_scale is not None:
                attn_scale = torch.Tensor([attn_scale]).float()
                extra_args.append(attn_scale)
                # print(f"Inference with lora scale: {attn_scale}")
            # TODO: warning
            # for key, value in cross_attention_kwargs.items():
                # print(f"Ignoring cross_attention_kwargs {key}, value: {value}")
        
        if isinstance(self.model, PytorchOpenVINOModel):
            timestep = timestep[None]

        down_block_additional_residuals, mid_block_additional_residual = get_dummy_unet_additional_residuals()
        extra_args.extend(down_block_additional_residuals)
        extra_args.append(mid_block_additional_residual)

        noise_pred = self.model(sample, timestep, encoder_hidden_states, *extra_args)
        if hasattr(noise_pred, "sample"):
            noise_pred = noise_pred.sample
        if isinstance(noise_pred, tuple):
            noise_pred = noise_pred[0]
        elif isinstance(noise_pred, dict):
            noise_pred = noise_pred["sample"]
        
        return UNet2DConditionOutput(sample=noise_pred)



