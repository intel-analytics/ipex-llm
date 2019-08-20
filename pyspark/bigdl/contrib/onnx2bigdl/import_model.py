import numpy as np
import onnx
from import_graph import BigDLGraph


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
		return self._graph
		
	def summary(self):
		print("IR version: " + str(self._ir_version))
		#print("Opset import: " + self._opset_import[0])
		# print(len(self._opset_import))
		# while len(self._opset_import) > 0:
		# 	curr_op = self._opset_import.pop()
		# 	print(curr_op.domain + " " + str(curr_op.version))
		# print(len(self._opset_import))
		print("Producer: " + self._producer_name + " " + self._producer_version)
		print("Domain: " + self._domain)
		print("Model version: " + str(self._model_version))
		print("Doc string: " + self._doc_string)
		

if __name__ == '__main__':
	model_path = "/home/leicongl/Workground/myData/models/onnx/alexnet.onnx"
	model = BigDLModel(model_path)
	model.summary()
	model.load_model()
	tensor = np.random.random([1, 3, 224, 224])
	out = model.forward(tensor)
	print(out)
	
