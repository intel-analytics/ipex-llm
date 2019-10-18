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

import onnx
from bigdl.nn.onnx.layer import *
from bigdl.nn.layer import Identity, Model
from .ops_mapping import _convert_map as convert_map
from .converter_utils import parse_node_attr, parse_tensor_data


class OnnxLoader(object):

    def load_model(self, file_path):
        model_proto = onnx.load_model(file_path)
        # self._ir_version = model_proto.ir_version
        # self._opset_import = model_proto.opset_import
        # self._producer_name = model_proto.producer_name
        # self._producer_version = model_proto.producer_version
        # self._domain = model_proto.domain
        # self._model_version = model_proto.model_version
        # self._doc_string = model_proto.doc_string
        graph_proto = model_proto.graph
        return self.load_graph(graph_proto)

    def load_graph(self, graph_proto):
        if not graph_proto:
            raise ValueError("Graph proto is required")

        input_nodes = list()
        output_nodes = list()
        tensor_map = dict()
        initialized_tensors = set()
        module_map = dict()
        root_nodes = list()
        dummy_root = Identity()()

        for tensor in graph_proto.initializer:
            if not tensor.name.strip():
                raise ValueError("Tensor's name is required")
            initialized_tensors.add(tensor.name)
            tensor_data = parse_tensor_data(tensor)
            tensor_map[tensor.name] = (tensor_data, tensor_data.shape)

        for gin in graph_proto.input:
            if gin.name not in initialized_tensors:
                input_nodes.append(gin.name)
                shape = tuple([dim.dim_value for dim in gin.type.tensor_type.shape.dim])
                module_map[gin.name] = Identity()(dummy_root)
                tensor_map[gin.name] = (None, shape)

        for gout in graph_proto.output:
            if gout.name not in initialized_tensors:
                output_nodes.append(gout.name)

        for node in graph_proto.node:
            name = node.name.strip()
            op_type = node.op_type
            inputs = [tensor_map[n] for n in node.input]
            outputs = node.output
            prev_modules = [module_map[n] for n in node.input if n not in initialized_tensors]
            attrs = parse_node_attr(node)

            if len(prev_modules) == 0:
                root_nodes.append((name, op_type))
                prev_modules = [dummy_root]

            bigdl_module, outputs_shape = self._make_module_from_onnx_node(op_type, inputs, prev_modules, attrs, outputs)

            assert len(outputs) == len(outputs_shape)

            for out, out_shape in zip(outputs, outputs_shape):
                module_map[out] = bigdl_module
                tensor_map[out] = (None, out_shape)

        in_modules = [module_map[m] for m in input_nodes]
        out_modules = [module_map[m] for m in output_nodes]
        model = Model([dummy_root], out_modules)

        return model

    def _make_module_from_onnx_node(self, op_type, inputs, prev_modules, attrs, outputs):
        module = None
        out_shapes = []
        if op_type in convert_map:
            module, out_shapes = convert_map[op_type](inputs, prev_modules, attrs, outputs)
        else:
            raise NotImplemented(op_type)
        return module, out_shapes


def load(model_path):
    loader = OnnxLoader()
    return loader.load_model(model_path)


def load_model_proto(model_proto):
    loader = OnnxLoader()
    return loader.load_graph(model_proto.graph)