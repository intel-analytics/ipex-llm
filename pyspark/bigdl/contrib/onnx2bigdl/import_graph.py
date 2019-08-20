
import numpy as np
from op_mapping import _convert_map as convert_map
from bigdl.nn.layer import *
 

class BigDLGraph(object):
	
	def __init__(self, graph_proto = None):
		self._graph = self.load_graph(graph_proto)
		self._inputs = [] # input module list
		self._outputs = [] # output module list
		self._tensors = {} # (tensor_name, tensor_val)
		self._modules = {} # (module_name, module_obj) pairs
		return

	def load_graph(self, graph_proto):
		if not graph_proto:
			return None

		meta = self._parse_graph_metadata(graph_proto) 
		tensor_set = set()

		for tensor in graph_proto.initializer:
			if not tensor.name.strip():
				raise ValueError("Tensor's name is required")
			tensor_set.add(tensor.name)
			tensor_data = self._parse_tensor_data(tensor)
			self._tensors[tensor.name] = (tensor_data, tensor_data.shape)
			
		for gin in graph_proto.input:
			if gin.name not in tensor_set:
				self._inputs.append(gin.name)
				input_tensor_data = meta['input_tensor_data']
				self._modules[gin.name] = Input()
				self._tensors[gin.name] = (None, input_tensor_data[gin.name])

		for gout in graph_proto.output:
			self._outputs.append(gout.name)

		for node in graph_proto.node:
			name = node.name.strip()
			op_type = node.op_type

			print("start", name, op_type)

			inputs = [self._tensors[n] for n in node.input]
			prev_modules = [self._modules[n] for n in node.input if n not in tensor_set]
			attrs = self._parse_node_attr(node)
			outputs = node.output

			bigdl_module, outputs_shape = self._make_module_from_onnx_node(op_type, inputs, prev_modules, attrs, outputs)

			assert len(outputs) == len(outputs_shape)

			for out, out_shape in zip(outputs, outputs_shape):
				print(out, bigdl_module)
				self._modules[out] = bigdl_module
				self._tensors[out] = (None, out_shape)

			print("end")

		in_modules = [self._modules[m] for m in meta['input_tensor_data'].keys()]
		out_modules = [self._modules[m] for m in self._outputs]

		print("Graph loaded.")
		model = Model(in_modules, out_modules)

		return model


	def _make_module_from_onnx_node(self, op_type, inputs, prev_modules, attrs, outputs):
		module = None
		out_shapes = []
		if op_type in convert_map:
			module, out_shapes = convert_map[op_type](inputs, prev_modules, attrs, outputs)
		else:
			raise NotImplemented(op_type)
		return module, out_shapes  


	def _parse_graph_metadata(self, graph_proto):
		_params = set()
		for tensor in graph_proto.initializer:
			_params.add(tensor.name)

		input_data = {}
		for gin in graph_proto.input:
			if gin.name not in _params:
				shape = [dim.dim_value for dim in gin.type.tensor_type.shape.dim]
				input_data[gin.name] = tuple(shape)

		output_data = {}
		for gout in graph_proto.output:
			shape = [dim.dim_value for dim in gout.type.tensor_type.shape.dim]
			output_data[gout.name] = tuple(shape)

		metadata = {
			'input_tensor_data' : input_data,
			'output_tensor_data' : output_data
		}
		
		return metadata 
		

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


if __name__ == '__main__':
	print()
