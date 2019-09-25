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

	def __init__(self, file_path=None):
		if file_path:
			self._model_proto = onnx.load_model(file_path)
		else:
			self._model_proto = None

	
	def get_model_proto(self):
		return self._model_proto

	def load_model(self, model_proto):
		self._model_proto = model_proto
		# self._ir_version = model_proto.ir_version
		# self._opset_import = model_proto.opset_import
		# self._producer_name = model_proto.producer_name
		# self._producer_version = model_proto.producer_version
		# self._domain = model_proto.domain
		# self._model_version = model_proto.model_version
		# self._doc_string = model_proto.doc_string
		bgraph = BigDLGraph()
		graph_proto = model_proto.graph
		return bgraph.load_graph(graph_proto)


def load_onnx(model_path):
	model = BigDLModel(model_path)
	model_proto = model.get_model_proto()
	bigdl_model = model.load_model(model_proto)
	return bigdl_model
