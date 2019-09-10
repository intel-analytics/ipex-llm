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
from .import_graph import BigDLGraph


class BigDLModel(object):

	def __init__(self, file_path):
		onnx_model = onnx.load_model(file_path)
		self._onnx_graph = onnx_model.graph
		self._ir_version = onnx_model.ir_version
		self._opset_import = onnx_model.opset_import
		self._producer_name = onnx_model.producer_name
		self._producer_version = onnx_model.producer_version
		self._domain = onnx_model.domain
		self._model_version = onnx_model.model_version
		self._doc_string = onnx_model.doc_string
		self._graph = None

	def load_model(self):
		bgraph = BigDLGraph()
		graph_proto = self._onnx_graph
		self._graph = bgraph.load_graph(graph_proto)
		self._debug_graph = bgraph
		return self._graph

	def summary(self):
		print("IR version: " + str(self._ir_version))
		print("Producer: " + self._producer_name + " " + self._producer_version)
		print("Domain: " + self._domain)
		print("Model version: " + str(self._model_version))
		print("Doc string: " + self._doc_string)


def load_onnx(model_path):
	model = BigDLModel(model_path)
	bmodel = model.load_model()
	return bmodel
