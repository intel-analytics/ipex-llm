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

# see https://github.com/ggerganov/llama.cpp/blob/master/convert.py
# and https://github.com/ggerganov/llama.cpp/blob/master/ggml-quants.c
# and https://github.com/ggerganov/llama.cpp/blob/master/llama.cpp

import struct
import functools
import torch
import numpy

from io import BufferedReader
from tqdm import tqdm
from bigdl.llm.utils.common import invalidInputError


class GGUFReader:
    def __init__(self, f: BufferedReader):
        self.f = f
        self.funcs = {
            0: self.read_u8,
            1: self.read_i8,
            2: self.read_u16,
            3: self.read_i16,
            4: self.read_u32,
            5: self.read_i32,
            6: self.read_f32,
            7: self.read_bool,
            8: self.read_str,
            9: self.read_array,
            10: self.read_u64,
            11: self.read_i64,
            12: self.read_f64,
        }

    def read_value(self):
        value_type = self.read_i32()
        value = self.funcs[value_type]()
        return value

    def read_bool(self):
        data = self.f.read(1)
        return struct.unpack("<?", data)[0]

    def read_i8(self):
        data = self.f.read(1)
        return struct.unpack("<b", data)[0]

    def read_u8(self):
        data = self.f.read(1)
        return struct.unpack("<B", data)[0]

    def read_i16(self):
        data = self.f.read(2)
        return struct.unpack("<h", data)[0]

    def read_u16(self):
        data = self.f.read(2)
        return struct.unpack("<H", data)[0]

    def read_i32(self):
        data = self.f.read(4)
        return struct.unpack("<i", data)[0]

    def read_u32(self):
        data = self.f.read(4)
        return struct.unpack("<I", data)[0]

    def read_i64(self):
        data = self.f.read(8)
        return struct.unpack("<q", data)[0]

    def read_u64(self):
        data = self.f.read(8)
        return struct.unpack("<Q", data)[0]

    def read_f32(self):
        data = self.f.read(4)
        return struct.unpack("<f", data)[0]

    def read_f64(self):
        data = self.f.read(8)
        return struct.unpack("<d", data)[0]

    def read_str(self):
        length = self.read_u64()
        data = self.f.read(length)
        return data.decode()

    def read_array(self):
        item_type = self.read_i32()
        item_num = self.read_u64()
        arr = [
            self.funcs[item_type]()
            for i in range(item_num)
        ]
        return arr


class GGUFHeader:
    size = 4 + 4 + 8 + 8

    def __init__(self, f: BufferedReader):
        data = f.read(GGUFHeader.size)

        magic = data[0:4].decode()
        invalidInputError(magic == "GGUF", "not a valid gguf file")

        version, n_tensors, n_kv = struct.unpack("<IQQ", data[4:])
        invalidInputError(version in [2, 3], "only gguf v2 and v3 is supported")

        self.magic = magic
        self.version = version
        self.n_tensors = n_tensors
        self.n_kv = n_kv


class GGUFConfig:
    def __init__(self, f: BufferedReader, header: GGUFHeader):
        self.config = {}

        reader = GGUFReader(f)
        for i in range(header.n_kv):
            key = reader.read_str()
            value = reader.read_value()
            self.config[key] = value


class GGUFTensorInfos:
    def __init__(self, f: BufferedReader, header: GGUFHeader, config: GGUFConfig):
        self.infos = []

        reader = GGUFReader(f)
        for i in range(header.n_tensors):
            name = reader.read_str()
            ndims = reader.read_u32()
            dims = [
                reader.read_u64()
                for i in range(ndims)
            ]
            dims = list(reversed(dims))
            qtype = reader.read_i32()
            offset = reader.read_u64()
            self.infos.append((name, ndims, dims, qtype, offset))

        alignment = config.config.get("general.alignment", 32)
        base_offset = (f.tell() + alignment - 1) // alignment * alignment
        self.base_offset = base_offset


class GGUFTensorLoader:
    def __init__(self, fpath: str, tensor_infos: GGUFTensorInfos):
        self.block_ne = {
            0: 1,       # f32
            1: 1,       # f16
            2: 32,      # q4_0
            3: 32,      # q4_1
            6: 32,      # q5_0
            7: 32,      # q5_1
            8: 32,      # q8_0
            9: 32,      # q8_1
            10: 256,    # q2_k
            11: 256,    # q3_k
            12: 256,    # q4_k
            13: 256,    # q5_k
            14: 256,    # q6_k
            15: 256,    # q8_k
            16: 1,      # i8
            17: 1,      # i16
            18: 1,      # i32
        }

        self.block_size = {
            0: 4,       # f32
            1: 2,       # f16
            2: 18,      # q4_0
            3: 20,      # q4_1
            6: 22,      # q5_0
            7: 24,      # q5_1
            8: 34,      # q8_0
            9: 40,      # q8_1
            10: 0,      # q2_k
            11: 0,      # q3_k
            12: 0,      # q4_k
            13: 0,      # q5_k
            14: 210,    # q6_k
            15: 0,      # q8_k
            16: 1,      # i8
            17: 2,      # i16
            18: 4,      # i32
        }

        self.convert_funcs = {
            0: self.convert_f32_tensor,         # f32
            1: self.convert_f16_tensor,         # f16
            2: self.convert_q4_0_tensor,        # q4_0
            3: self.convert_q4_1_tensor,        # q4_1
            6: self.convert_q5_0_tensor,        # q5_0
            7: self.convert_q5_1_tensor,        # q5_1
            8: self.convert_q8_0_tensor,        # q8_0
            9: self.convert_unknown_tensor,     # q8_1
            10: self.convert_unknown_tensor,    # q2_k
            11: self.convert_unknown_tensor,    # q3_k
            12: self.convert_unknown_tensor,    # q4_k
            13: self.convert_unknown_tensor,    # q5_k
            14: self.convert_q6_k_tensor,       # q6_k
            15: self.convert_unknown_tensor,    # q8_k
            16: self.convert_unknown_tensor,    # i8
            17: self.convert_unknown_tensor,    # i16
            18: self.convert_unknown_tensor,    # i32
        }

        self.fpath = fpath
        self.infos = tensor_infos.infos
        self.base_offset = tensor_infos.base_offset

    def __iter__(self):
        with open(self.fpath, 'rb') as f:
            for name, ndims, dims, qtype, offset in tqdm(self.infos, desc="Loading gguf tensors"):
                total_ne = functools.reduce(lambda x, y: x * y, dims)
                invalidInputError(total_ne % self.block_ne[qtype] == 0,
                                  f"wrong elements num: {dims}")

                size = total_ne // self.block_ne[qtype] * self.block_size[qtype]
                invalidInputError(size != 0, f"unsupported quantize type: {qtype}")

                offset += self.base_offset
                f.seek(offset)
                data = f.read(size)
                arr = numpy.frombuffer(data, dtype=numpy.uint8).copy()
                tensor = torch.from_numpy(arr)
                tensor = self.convert_funcs[qtype](tensor, size, ndims, dims)
                yield name, tensor

    def load_while_process(self, process):
        with open(self.fpath, 'rb') as f:
            for name, ndims, dims, qtype, offset in tqdm(self.infos, desc="Loading gguf tensors"):
                total_ne = functools.reduce(lambda x, y: x * y, dims)
                invalidInputError(total_ne % self.block_ne[qtype] == 0,
                                  f"wrong elements num: {dims}")

                size = total_ne // self.block_ne[qtype] * self.block_size[qtype]
                invalidInputError(size != 0, f"unsupported quantize type: {qtype}")

                offset += self.base_offset
                f.seek(offset)
                data = f.read(size)
                arr = numpy.frombuffer(data, dtype=numpy.uint8)
                tensor = torch.from_numpy(arr)
                tensor = self.convert_funcs[qtype](tensor, size, ndims, dims)
                process(name, tensor)

    def convert_f32_tensor(self, tensor: torch.Tensor, size: int, ndims: int, dims: int):
        return tensor.view(torch.float)

    def convert_f16_tensor(self, tensor: torch.Tensor, size: int, ndims: int, dims: int):
        return tensor.view(torch.half).reshape(dims)

    def convert_q4_0_tensor(self, tensor: torch.Tensor, size: int, ndims: int, dims: int):
        # see https://github.com/ggerganov/llama.cpp/blob
        # /8e672efe632bb6a7333964a255c4b96f018b9a65/ggml-quants.c#L1074

        block_size = self.block_size[2]
        tensor = tensor.reshape((-1, block_size))
        scales, data = tensor[:, :2], tensor[:, 2:]
        scales = scales.view(torch.half)
        data = torch.cat([data & 0xF, data >> 4], dim=-1).view(torch.int8) - 8
        result = (data * scales).reshape(dims)
        return result

    def convert_q4_1_tensor(self, tensor: torch.Tensor, size: int, ndims: int, dims: int):
        # see https://github.com/ggerganov/llama.cpp/blob
        # /b38a16dfcff88d547f78f52d1bea31b84a05aff7/ggml-quants.c#L1094

        block_size = self.block_size[3]
        tensor = tensor.reshape((-1, block_size))
        scales, base, data = tensor[:, :2], tensor[:, 2:4], tensor[:, 4:]
        scales = scales.view(torch.half)
        base = base.view(torch.half)
        data = torch.cat([data & 0xF, data >> 4], dim=-1)
        result = (data * scales + base).reshape(dims)
        return result

    def convert_q5_0_tensor(self, tensor: torch.Tensor, size: int, ndims: int, dims: int):
        # see https://github.com/ggerganov/llama.cpp/blob
        # /b38a16dfcff88d547f78f52d1bea31b84a05aff7/ggml-quants.c#L1115

        block_size = self.block_size[6]
        tensor = tensor.reshape((-1, block_size))
        scales, hdata, ldata = tensor[:, :2], tensor[:, 2:6], tensor[:, 6:]
        scales = scales.view(torch.half)
        # hdata = hdata.view(torch.int)
        hdata = hdata.clone().view(torch.int)   # clone hdata to fix memory address alignment
        shift = torch.arange(0, 32, 1)
        hdata = (((hdata.expand(-1, 32) >> shift) << 4) & 0x10).byte()
        ldata = torch.cat([ldata & 0xF, ldata >> 4], dim=-1)
        data = (hdata | ldata).view(torch.int8) - 16
        result = (data * scales).reshape(dims)
        return result

    def convert_q5_1_tensor(self, tensor: torch.Tensor, size: int, ndims: int, dims: int):
        # https://github.com/ggerganov/llama.cpp/blob
        # /b38a16dfcff88d547f78f52d1bea31b84a05aff7/ggml-quants.c#L1141

        block_size = self.block_size[7]
        tensor = tensor.reshape((-1, block_size))
        scales, base, hdata, ldata = tensor[:, :2], tensor[:, 2:4], tensor[:, 4:8], tensor[:, 8:]
        scales = scales.view(torch.half)
        base = base.view(torch.half)
        hdata = hdata.view(torch.int)
        shift = torch.arange(0, 32, 1)
        hdata = (((hdata.expand(-1, 32) >> shift) << 4) & 0x10).byte()
        ldata = torch.cat([ldata & 0xF, ldata >> 4], dim=-1)
        data = hdata | ldata
        result = (data * scales + base).reshape(dims)
        return result

    def convert_q8_0_tensor(self, tensor: torch.Tensor, size: int, ndims: int, dims: int):
        # https://github.com/ggerganov/llama.cpp/blob
        # /b38a16dfcff88d547f78f52d1bea31b84a05aff7/ggml-quants.c#L1168

        block_size = self.block_size[8]
        tensor = tensor.reshape((-1, block_size))
        scales, data = tensor[:, :2], tensor[:, 2:]
        scales = scales.view(torch.half)
        data = data.view(torch.int8)
        result = (data * scales).reshape(dims)
        return result

    def convert_q6_k_tensor(self, tensor: torch.Tensor, size: int, ndims: int, dims: int):
        # see https://github.com/ggerganov/llama.cpp/blob
        # /8e672efe632bb6a7333964a255c4b96f018b9a65/ggml-quants.c#L2263

        block_size = self.block_size[14]
        tensor = tensor.reshape((-1, block_size))

        ql, qh, scales, d = (tensor[:, :128], tensor[:, 128:192],
                             tensor[:, 192:208], tensor[:, 208:])
        data_0 = (ql[:, 00:32] & 0xF) | ((qh[:, :32] & 0B00000011) << 4)
        data_1 = (ql[:, 32:64] & 0xF) | ((qh[:, :32] & 0B00001100) << 2)
        data_2 = (ql[:, 00:32] >> 4) | ((qh[:, :32] & 0B00110000) >> 0)
        data_3 = (ql[:, 32:64] >> 4) | ((qh[:, :32] & 0B11000000) >> 2)
        data_4 = (ql[:, 64:96] & 0xF) | ((qh[:, 32:64] & 0B00000011) << 4)
        data_5 = (ql[:, 96:128] & 0xF) | ((qh[:, 32:64] & 0B00001100) << 2)
        data_6 = (ql[:, 64:96] >> 4) | ((qh[:, 32:64] & 0B00110000) >> 0)
        data_7 = (ql[:, 96:128] >> 4) | ((qh[:, 32:64] & 0B11000000) >> 2)
        data = torch.cat([data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7],
                         dim=-1).view(torch.int8) - 32
        result = data * d.view(torch.half)

        result = result.reshape((-1, 16, 16)) * scales.view(torch.int8).reshape((-1, 16, 1))
        result = result.reshape(dims)
        return result

    def convert_unknown_tensor(self, tensor: torch.Tensor, size: int, ndims: int, dims: int):
        invalidInputError(False, "Unsupported qtype")


class GGUFFileLoader:
    def __init__(self, fpath: str):
        with open(fpath, 'rb') as f:
            header = GGUFHeader(f)
            config = GGUFConfig(f, header)
            tensor_infos = GGUFTensorInfos(f, header, config)
            tensor_loader = GGUFTensorLoader(fpath, tensor_infos)

        self.header = header
        self.config = config.config
        self.tensor_loader = tensor_loader

    def tensors(self, dtype: torch.dtype = torch.float):
        return {
            name: value.to(dtype=dtype)
            for name, value in self.tensor_loader
        }

    def tensors_iter(self):
        return self.tensor_loader

    def tokenizer_pieces(self):
        from transformers.convert_slow_tokenizer import import_protobuf
        spm_pb2 = import_protobuf("Failed to import protobuf")

        tokens = self.config['tokenizer.ggml.tokens']
        token_types = self.config['tokenizer.ggml.token_type']
        merges = None
        if 'tokenizer.ggml.scores' in self.config:
            scores = self.config['tokenizer.ggml.scores']
        elif self.config['tokenizer.ggml.model'] == "gpt2":
            merges = self.config['tokenizer.ggml.merges']
            scores = list(range(len(tokens)))
        else:
            invalidInputError(False, "Invalid configuration: 'scores' is not provided.")

        pieces = [
            spm_pb2.ModelProto.SentencePiece(
                piece=token,
                score=score,
                type=token_type,
            )
            for token, score, token_type in tqdm(
                zip(tokens, scores, token_types),
                "Loading gguf vocab"
            )
        ]

        if merges is not None:
            return pieces, merges
        else:
            return pieces
