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


from openvino.runtime import Core, serialize
import os
from ipex_llm.transformers.npu_models.mp_models_base import LLMBaseNNFactory
from typing import Sequence
from intel_npu_acceleration_library.backend.factory import NNFactory
import numpy as np
import torch


def update_names_of_IR_and_export_blob(model, model_name, dir, compile_blob=True, keep_ir=True,
                                       npu_dpu_groups=None):
    xml_path = os.path.join(dir, model_name + ".xml")
    bin_path = os.path.join(dir, model_name + ".bin")
    model.serialize(xml_path, bin_path)
    new_ir_path = os.path.join(dir, model_name + "_new.xml")
    new_bin_path = os.path.join(dir, model_name + "_new.bin")
    blob_path = os.path.join(dir, model_name + ".blob")

    core = Core()
    core.set_property("NPU", {"NPU_COMPILATION_MODE_PARAMS":
                              "compute-layers-with-higher-precision=Sqrt,Power,ReduceMean,Add"})
    core.set_property("NPU", {"PERFORMANCE_HINT": "LATENCY"})
    if (
        npu_dpu_groups is not None
        and os.environ.get("IPEX_LLM_NPU_DISABLE_COMPILE_OPT", "0") != "1"
    ):
        core.set_property("NPU", {"NPU_DPU_GROUPS": str(npu_dpu_groups)})

    model = core.read_model(xml_path)
    inputs = model.inputs
    for idx, input in enumerate(inputs):
        if len(input.names) == 0:
            model.inputs[idx].set_names({f"input_{idx}"})
    outputs = model.outputs
    for idx, input in enumerate(outputs):
        if len(input.names) == 0:
            model.outputs[idx].set_names({f"output_{idx}"})
    # rewrite this model to a new IR path
    if new_ir_path is not None:
        serialize(model, new_ir_path)

    if compile_blob:
        compiledModel = core.compile_model(model, device_name="NPU")
        model_stream = compiledModel.export_model()
        with open(blob_path, 'wb') as f:
            f.write(model_stream)

    os.remove(xml_path)

    if not keep_ir:
        os.remove(new_ir_path)
        os.remove(new_bin_path)

    return blob_path


class LowBitLLMLMHead(LLMBaseNNFactory):
    def __init__(
        self,
        hidden_shape: Sequence[int],
        num_heads: int,
        rms_norm_eps: float,
        model_norm_weight,
        vocab_size: int,
        mode: str = "decode",
        dtype: np.dtype = np.int8,
        max_seq_len: int = 1024,
        transpose_value: bool = False,
        profile: bool = False,
        device: str = "NPU",
        n_splits: int = 1,
        group_size: int = 0,
        asym: bool = False
    ):
        super().__init__(max_seq_len=max_seq_len,
                         transpose_value=transpose_value,
                         dtype=dtype,
                         profile=profile,
                         device=device)
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.batch_size, self.seq_len, self.hidden_size = hidden_shape
        self.mode = mode
        self.rms_norm_eps = rms_norm_eps
        self.transpose_value = transpose_value
        self.vocab_size = vocab_size

        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads

        # define input, the order self.parameter matters
        if n_splits == 1:
            input = self.create_input_op((self.batch_size, self.seq_len, self.hidden_size))
        else:
            input = self.create_input_op((1, self.batch_size, self.hidden_size))

        hidden_states = input

        # model norm and lm head
        model_norm_weight = self.constant(model_norm_weight)
        hidden_states = self.layer_norm(hidden_states, model_norm_weight)

        hidden_states = self.linear(
            hidden_states, self.vocab_size, self.hidden_size, bias=False, wt_dtype=self.dtype,
            n_splits=n_splits,
            scale_factor=(group_size == 0),
            asym=asym
        )

        # define outputs
        hidden_states = self.convert_to_fp32(hidden_states)

        print("start compiling")
        self.compile()


class LLMEmbedding(NNFactory):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        embedding_weight,
        padding_idx,
        dtype,  # fp16
        input_length: int = 1,
        device: str = "NPU",
    ):
        super().__init__(False, device)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.dtype = dtype

        # define input
        weight = self.constant(embedding_weight)
        input = self.parameter((1, input_length), dtype=np.int32)

        if padding_idx == -1:
            padding_idx += vocab_size

        axis_node = self.constant(np.array([0], dtype=np.int64))
        if padding_idx is not None:
            masked_embeddings = np.ones(weight.shape, dtype=np.float16)
            masked_embeddings[padding_idx, :] = 0.0  # mask

            node_mask = self.constant(masked_embeddings)
            node_masked_w = self.eltwise_mul(weight, node_mask)
            res = self.gather(node_masked_w, input, axis_node, 0)
        else:
            res = self.gather(weight, input, axis_node, 0)

        # define outputs
        res = self.convert_to_fp16(res)

        print("start compiling")
        self.compile()


def obtain_weight_from_single_layer(attn_layer, mlp_layer):
    weights = []
    if hasattr(attn_layer, "q_proj_dq_list"):
        for layer_list in [attn_layer.q_proj_dq_list, attn_layer.k_proj_dq_list,
                           attn_layer.v_proj_dq_list, attn_layer.o_proj_dq_list,
                           mlp_layer.gate_proj_dq_list, mlp_layer.up_proj_dq_list,
                           mlp_layer.down_proj_dq_list]:
            l_weights = []
            scales = []
            zeros = []
            for l in layer_list:
                l_weights.append(l.weight)
                scales.append(l.scale)
                if l.zero is not None:
                    zeros.append(l.zero)
            if len(zeros):
                weights.append((torch.stack(l_weights, axis=0), torch.stack(scales, axis=0),
                                torch.stack(zeros, axis=0)))
            else:
                weights.append((torch.stack(l_weights, axis=0),
                                torch.stack(scales, axis=0)))
    else:
        for layer in [attn_layer.q_proj, attn_layer.k_proj,
                      attn_layer.v_proj, attn_layer.o_proj,
                      mlp_layer.gate_proj, mlp_layer.up_proj,
                      mlp_layer.down_proj]:
            if layer.zero is not None:
                weights.append((layer.weight, layer.scale, layer.zero))
            else:
                weights.append((layer.weight, layer.scale))
    return weights


def obtain_qkv_bias_from_single_layer(attn_layer):
    if hasattr(attn_layer, "q_proj_dq_list"):
        q_bias = attn_layer.q_proj_dq_list.q_proj_dq_0.bias.to(torch.float16)
        k_bias = attn_layer.k_proj_dq_list.k_proj_dq_0.bias.to(torch.float16)
        v_bias = attn_layer.v_proj_dq_list.v_proj_dq_0.bias.to(torch.float16)
    else:
        q_bias = attn_layer.q_proj.bias.to(torch.float16)
        k_bias = attn_layer.k_proj.bias.to(torch.float16)
        v_bias = attn_layer.v_proj.bias.to(torch.float16)
    return q_bias, k_bias, v_bias
