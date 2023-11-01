from bigdl.llm.transformers.low_bit_linear import ggml_convert_qtype
from bigdl.llm.transformers.low_bit_linear import ggml_q_format_convet_cpu2xpu
from bigdl.llm.ggml.quantize import ggml_tensor_qtype
import torch
import intel_extension_for_pytorch as ipex

NF4_MAP = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
-0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0]
# nf4_map = torch.tensor(NF4_MAP, dtype=torch.bfloat16, device="hpu").reshape(-1, 1)
nf4_map = None
from operator import mul
from functools import reduce
def torch_dequant_nf4(quant_weight, original_shape=(4096, 4096)):
    num_elements = reduce(mul, original_shape, 1)
    quants = quant_weight[:num_elements//2]
    scales = get_scales(quant_weight, num_elements)
    quants = quants.reshape(-1, 32)
    quants_0 = quants.bitwise_right_shift(4)
    quants_1 = quants.bitwise_and(0x0f)

    global nf4_map
    if nf4_map is None:
        nf4_map = torch.tensor(NF4_MAP, dtype=torch.bfloat16, device=quant_weight.device).reshape(-1, 1)

    dequants_0 = torch.nn.functional.embedding(quants_0.int(), nf4_map)
    dequants_1 = torch.nn.functional.embedding(quants_1.int(), nf4_map)

    # print(dequants_0.shape)
    # print(scales.shape)
    dequants_0 = dequants_0 * scales.reshape(-1, 1, 1)
    dequants_1 = dequants_1 * scales.reshape(-1, 1, 1)

    return torch.stack([dequants_0, dequants_1], dim=1).reshape(original_shape)

def get_scales(quant_weight, num_elements):
    scales = quant_weight[num_elements//2:]

    fp16_tensor = torch.empty(0, dtype=torch.float16, device=quant_weight.device)
    fp16_tensor.set_(scales.untyped_storage(),
                     scales.storage_offset()//2,
                     size=(num_elements//64,),
                     stride=(1,))
    # assert scales.storage_offset() == num_elements // 2
    # assert num_elements // 64 * 2== (quant_weight.numel() - num_elements // 2), \
    #     f"{num_elements // 64 * 2} != {(quant_weight.numel() - num_elements // 2)}"
    return fp16_tensor