from typing import Tuple
import torch
from typing import Union, Callable, Any, Optional

def module_optimization(func: Callable) -> torch.nn.Module:
    """Optimize recursively a torch.nn.Module with a specific function.

    The function `func` get called recursively to every module in the network.

    Args:
        func (Callable): optimization function

    Returns:
        torch.nn.Module: optimized module
    """

    def wrapper(model: torch.nn.Module, *args: Any, **kwargs: Any):
        """Recursively apply the optimization function.

        Args:
            model (torch.nn.Module): original module
            args (Any): positional arguments
            kwargs (Any): keyword arguments

        """
        for name, layer in model.named_children():
            new_layer = func(name, layer, *args, **kwargs)
            if new_layer:
                model.add_module(name, new_layer)
                wrapper(new_layer, *args, **kwargs)
            else:
                wrapper(layer, *args, **kwargs)

    return wrapper

def quantize_tensor(
    weight: torch.Tensor, min_max_range: Tuple[int, int] = (-128, 127)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a fp16 tensor symmetrically.

    Produces a quantize tensor (same shape, dtype == `torch.int8`) and a scale tensor (dtype == `torch.float16)
    The quantization equation is the following W = S * W_q

    Args:
        weight (torch.Tensor): The tensor to quantize
        min_max_range (Tuple[int, int]): The min and max range for the quantized tensor. Defaults to (-128, 127).

    Raises:
        RuntimeError: Error in the quantization step

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Quantized tensor and scale
    """
    scale = torch.max(torch.abs(weight), dim=-1).values

    # if any of the elements are zeros set the scale to the max value
    if torch.any(scale == 0):
        scale = torch.ones_like(scale) * torch.max(torch.abs(weight))

    # Compute scale and zero point
    scale = (scale / max(min_max_range)).to(torch.float16).view(-1, 1)

    weights_quant = torch.floor(weight / scale)

    if not (
        torch.max(weights_quant) <= max(min_max_range)
        and torch.min(weights_quant) >= min(min_max_range)
    ):
        raise RuntimeError(
            f"Quantization error: range of quantized weghts = {(torch.min(weights_quant), torch.max(weights_quant))} instead of ({min_max_range})"
        )
    return weights_quant.to(torch.int8), scale.view(1, -1)


def compress_to_i4(weights: torch.Tensor) -> torch.Tensor:
    """
    Compresses a given tensor to 4-bit representation.

    Args:
        weights (torch.Tensor): The input tensor to be compressed.

    Returns:
        torch.Tensor: The compressed tensor with 4-bit representation.
    """
    compressed_weights = torch.zeros(
        (weights.shape[0], weights.shape[1] // 2), dtype=torch.uint8
    )
    for i in range(weights.shape[1] // 2):
        compressed_weights[:, i] = (weights[:, 2 * i] & 0x0F) | (
            ((weights[:, 2 * i + 1] & 0x0F) << 4) & 0xF0
        )
    return compressed_weights


class QuantizedLinear(torch.nn.Module):
    """Torch Quantized Linear operation NPU backend."""

    def __init__(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        """Initialize the QuantizedLinear class.

        Args:
            weight (torch.Tensor): Linear operation weight
            scale (torch.Tensor): Quantization scale
            bias (Optional[torch.Tensor], optional): Linear operation optional bias. Defaults to None.

        Raises:
            RuntimeError: Quantized weight must be in torch.int8 format
        """
        super().__init__()

        self.weight = weight
        if self.weight.dtype not in (torch.int8, torch.uint8):
            raise RuntimeError(
                f"Quantized weight must be in torch.(u)int8 dtype instead of {self.weight.dtype}"
            )
        self.scale = scale
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Torch module forward method.

        Args:
            x (torch.Tensor): Input tensor

        Raises:
            RuntimeError: Training is not supported for QuantizedLinear layer. Use `.eval()` to do inference only

        Returns:
            torch.Tensor: result
        """
        # print(x.shape)
        # print(self.weight.shape)
        # print(self.scale.shape)
        
        x_2d = x.view(-1, x.size(-1))
        out = (x_2d @ self.weight.T.to(torch.float16)) * self.scale
        
        out = out.view(x.shape[:-1] + (out.size(-1),))
        if self.bias is None:
            return out
        return out + self.bias
    
    @staticmethod
    def fromTensor(
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        qtype="sym_int8"
    ):
        """Generate a NPU Linear layer from a torch one.

        Args:
            weight (torch.Tensor): the original weight tensor
            bias (Optional[torch.Tensor]): the original bias tensor
            dtype (torch.dtype): the desired datatype

        Raises:
            RuntimeError: Quantized Linear requires input_channel to be a multiple of 8

        Returns:
            Union[Linear, QuantizedLinear]: A NPU linear layer
        """
        if qtype == "sym_int8":
            weights_quant, scale = quantize_tensor(weight, (-128, 127))
        elif qtype == "sym_int4":
            weights_quant, scale = quantize_tensor(weight, (-8, 7))
            # weights_quant = compress_to_i4(weights_quant)
        return QuantizedLinear(weights_quant, scale, bias)


@module_optimization
def lower_linear(
    name: str, layer: torch.nn.Module, qtype: str
) -> Union[torch.nn.Module, None]:
    """Lower torch.nn.Linear layer to NPU equivalent operators.

    Args:
        name (str): Layer name
        layer (torch.nn.Module): Original torch.nn.Linear module
        dtype (torch.dtype): Target datatype

    Returns:
        Union[torch.nn.Module, None]: Return the new NPU operator or None
    """
    if isinstance(layer, torch.nn.Linear):
        return QuantizedLinear.fromTensor(layer.weight, getattr(layer, "bias", None), qtype)
    return None