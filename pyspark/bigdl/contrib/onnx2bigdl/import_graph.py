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

import numpy as np
from bigdl.nn.onnx.layer import *
from bigdl.nn.layer import Identity, Model

from .ops_mapping import _convert_map as convert_map


class BigDLGraph(object):
	def __init__(self, graph_proto = None):
		self._inputs = list()   # input module list
		self._outputs = list()  # output module list
		self._tensors = dict()  # (tensor_name, tensor_val)
		self._modules = dict()  # (module_name, module_obj) pairs
		self._root = list()
		self._dummy_root = Identity()()
		self._graph = self.load_graph(graph_proto)
		return

	def load_graph(self, graph_proto):
		if not graph_proto:
			return None
		tensor_set = set()

		for tensor in graph_proto.initializer:
			if not tensor.name.strip():
				raise ValueError("Tensor's name is required")
			tensor_set.add(tensor.name)
			tensor_data = self._parse_tensor_data(tensor)
			self._modules[tensor.name] = Constant(tensor_data)(self._dummy_root)
			self._tensors[tensor.name] = (tensor_data, tensor_data.shape)
			
		for gin in graph_proto.input:
			if gin.name not in tensor_set:
				self._inputs.append(gin.name)
				shape = tuple([dim.dim_value for dim in gin.type.tensor_type.shape.dim])
				# input_tensor_data = meta['input_tensor_data']
				self._modules[gin.name] = Identity()(self._dummy_root)  # Input()
				self._tensors[gin.name] = (None, shape)

		for gout in graph_proto.output:
			if gout.name not in tensor_set:
				self._outputs.append(gout.name)

		for node in graph_proto.node:
			name = node.name.strip()
			op_type = node.op_type
			inputs = [self._tensors[n] for n in node.input]
			prev_modules = [self._modules[n] for n in node.input]
			attrs = self._parse_node_attr(node)
			outputs = node.output

			if len(prev_modules) == 0:
				self._root.append((name, op_type))
				prev_modules = [self._dummy_root]

			bigdl_module, outputs_shape = self._make_module_from_onnx_node(op_type, inputs, prev_modules, attrs, outputs)

			assert len(outputs) == len(outputs_shape)

			for out, out_shape in zip(outputs, outputs_shape):
				self._modules[out] = bigdl_module
				self._tensors[out] = (None, out_shape)

		in_modules = [self._modules[m] for m in self._inputs]
		out_modules = [self._modules[m] for m in self._outputs]
		model = Model([self._dummy_root], out_modules)

		return model

	def _make_module_from_onnx_node(self, op_type, inputs, prev_modules, attrs, outputs):
		module = None
		out_shapes = []
		if op_type in convert_map:
			module, out_shapes = convert_map[op_type](inputs, prev_modules, attrs, outputs)
		else:
			raise NotImplemented(op_type)
		return module, out_shapes

	def _parse_tensor_data(self, tensor_proto):
		try:
			from onnx.numpy_helper import to_array
		except ImportError:
			raise ImportError("Onnx and protobuf need to be installed.")
		if len(tuple(tensor_proto.dims)) > 0:
			np_array = to_array(tensor_proto).reshape(tuple(tensor_proto.dims))
		else:
			# If it is a scalar tensor
			np_array = np.array([to_array(tensor_proto)])
		return np_array

	def _parse_node_attr(self, node_proto):
		attrs = {}
		attr_proto = node_proto.attribute

		for attr in attr_proto:
			for field in ['f', 'i', 's']:
				if attr.HasField(field):
					attrs[attr.name] = getattr(attr, field)
					# Needed for supporting python version > 3.5
					if isinstance(attrs[attr.name], bytes):
						attrs[attr.name] = attrs[attr.name].decode(encoding='utf-8')
			for field in ['floats', 'ints', 'strings']:
				if list(getattr(attr, field)):
					assert attr.name not in attrs, "Only one type of attr is allowed"
					attrs[attr.name] = tuple(getattr(attr, field))

			for field in ['t', 'g']:
				if attr.HasField(field):
					attrs[attr.name] = getattr(attr, field)
			for field in ['tensors', 'graphs']:
				if list(getattr(attr, field)):
					raise NotImplementedError()
			if attr.name not in attrs:
				raise ValueError("Cannot parse attribute: \n{}\n.".format(attr))

		return attrs
