# Test data for convolution
convolutionDefinition = """
name : "ConvolutionTest"
input : "data"
input_shape {dim:1 dim :3 dim :5 dim :5}
layer {
 name : "convolution"
 type : "Convolution"
 bottom : "data"
 top : "convolution"
 convolution_param {
  num_output : 4
  kernel_size: 2
  weight_filler {
   type: "xavier"
 }
  bias_filler {
  type: "gaussian"
  std: 0.02
   }
  }
 }
"""
convolutionShapes = [{"data": (1, 3, 5, 5)}]
convolutionName = "convolution"
# End layer definitions
testlayers = []


class caffe_test_layer():
    def __init__(self, name, definition, shapes):
        self.name = name
        self.definition = definition
        self.shapes = shapes


def registerTestLayer(name, definition, shapes):
    layer = caffe_test_layer(name, definition, shapes)
    testlayers.append(layer)
registerTestLayer(convolutionName, convolutionDefinition, convolutionShapes)
