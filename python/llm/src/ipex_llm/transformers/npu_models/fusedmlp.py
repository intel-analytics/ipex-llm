from intel_npu_acceleration_library.backend.factory import NNFactory
from intel_npu_acceleration_library.backend.runtime import set_contiguous, record_function, adapt_output_tensor, _model_cache
from typing import Optional, Sequence, List, Union, Any
from functools import partial
from collections import deque
import numpy as np
import torch
import uuid


class QuantizedMLP(NNFactory):
    """Quantized Linear class, computing a matrix matrix multiplication with weights prefetching."""

    def __init__(
        self,
        input_shape: Sequence[int],
        intermediate_size: int,
        activation: str = "swiglu",
        bias: Optional[bool] = False,
        dtype: np.dtype = np.int8,
        profile: bool = False,
        device: str = "NPU",
        **additional_args
    ):
        """Initialize the Linear class.

        Args:
            input_shape (Sequence[int]): input shape channels
            intermediate_size (int): intermediate_size
            activation (str): activation function to use
            bias (Optional[bool], optional): Enable/Disable bias. Defaults to False.
            profile (bool): Enable/Disable profiling. Defaults to False.
            device (str): Target device, default to "NPU".
            additional_args: additional arguments
        """
        super().__init__(profile, device)
        self.intermediate_size = intermediate_size
        self.batch, self.hidden_size = input_shape
        input = self.parameter((self.batch, self.hidden_size))

        mm1 = self.linear(input, self.intermediate_size, self.hidden_size, bias=bias, wt_dtype=dtype)

        if activation == "swiglu":
            mm2 = self.linear(input, self.intermediate_size, self.hidden_size, bias=bias, wt_dtype=dtype)  # type: ignore[attr-defined]
            mm1 = self.eltwise_mul(self.swish(mm1), mm2)  # type: ignore[attr-defined]
        elif activation == "clamp":
            atc_fn = getattr(self, activation)
            mm1 = atc_fn(mm1, additional_args.get("min"), additional_args.get("max"))
        elif activation == "elu":
            atc_fn = getattr(self, activation)
            mm1 = atc_fn(mm1, additional_args.get("alpha", 1.0))
        elif activation == "grn":
            atc_fn = getattr(self, activation)
            mm1 = atc_fn(mm1, additional_args.get("grn_bias"))
        else:
            atc_fn = getattr(self, activation)
            mm1 = atc_fn(mm1)

        _ = self.linear(mm1, self.hidden_size, self.intermediate_size, bias=bias, wt_dtype=dtype)
        self.compile()


class FusedLlamaQuantizedMLP(torch.nn.Module):
    """LLAMA MLP operation NPU backend."""

    def __init__(
        self,
        parameters: List[torch.Tensor],
    ):
        """Initialize LLAMA MLP operation.

        Args:
            parameters (List[torch.Tensor]): model weights
        """
        super().__init__()
        self.op_parameters = parameters
        self.op_id = str(uuid.uuid4())
        np_dtype = np.float16
        if isinstance(parameters[0], tuple):  # from QuantizedLinear
            np_dtype = np.int8 if parameters[0][0].dtype == torch.int8 else np.uint8
            intermediate_size, _ = parameters[0][0].shape
        else:
            intermediate_size, _ = parameters[0].shape
        self.backend_cls = partial(QuantizedMLP, intermediate_size=intermediate_size, dtype=np_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Torch module forward method.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: result
        """
        original_shape = x.shape
        if len(x.shape) > 2:
            x = x.view([-1, x.shape[-1]])
        output = run_factory(x, self.op_parameters, self.backend_cls, self.op_id)
        return output.view(original_shape)


# TODO: separate it into a single file
@torch.no_grad()
def run_factory(
    x: Union[torch.Tensor, List[torch.Tensor]],
    weights: List[torch.Tensor],
    backend_cls: Any,
    op_id: Optional[str] = None,
) -> torch.Tensor:
    """Run a factory operation. Depending on the datatype of the weights it runs a float or quantized operation.

    Args:
        x (Union[torch.Tensor, List[torch.Tensor]]): Activation tensor(s). Its dtype must be torch.float16
        weights (torch.Tensor): Weights tensor.  Its dtype can be torch.float16 or torch.int8
        backend_cls (Any): Backend class to run
        op_id (Optional[str], optional): Operation ID. Defaults to None.

    Returns:
        torch.Tensor: result
    """
    global _model_cache

    # Use or not op_id depending on the class used
    op_kwargs = {"op_id": op_id} if op_id else {}

    if not isinstance(x, (list, tuple)):
        x = [x]

    # Reshape input
    input_dtype = x[0].dtype
    x_np = [set_contiguous(elem).to(torch.float16).numpy() for elem in x]
    op_args = []
    op_args_flatten = []
    for w in weights:
        if isinstance(w, tuple): # from QuantizedLinear
            op_args.append((set_contiguous(w[0]).numpy(), set_contiguous(w[1]).numpy()))
            op_args_flatten.append(op_args[-1][0])
            op_args_flatten.append(op_args[-1][1])
        else:
            op_args.append(set_contiguous(w).numpy())
            op_args_flatten.append(op_args[-1])

    shape_dtype_signature = "_".join(
        ["_".join(str(dim) for dim in t.shape) + f"_{t.dtype}" for t in x_np + op_args_flatten]
    )
    key = f"{backend_cls.func.__name__}_{shape_dtype_signature}"
    models = _model_cache.get(key, None)

    input_shapes = [elem.shape for elem in x_np]
    if models is None:
        _model_cache[key] = deque([backend_cls(*input_shapes) for i in range(4)])
    elif len(models) < 1:
        _model_cache[key].append(backend_cls(*input_shapes))
    else:
        _model_cache[key].rotate(1)

    # Get the model
    model = _model_cache[key][0]

    with record_function(f"npu_factory_mul_{key}"):
        ret = model.run(*x_np, *op_args, **op_kwargs)

    return adapt_output_tensor(ret, ret.shape, input_dtype)
