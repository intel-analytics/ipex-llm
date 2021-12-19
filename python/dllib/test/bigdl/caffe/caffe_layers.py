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


# Test data for Relu
reluDefinition = """
name : "ReluTest"
input : "data"
input_shape{dim:2 dim :2}
 layer {
  name: "relu"
  type: "ReLU"
  bottom: "data"
  top: "relu"
}
"""
reluShapes = [{"data": (2, 2)}]
reluName = "relu"


# Test Data for SpatialCrossMapLRN
crossMapLrnDefinition = """
name : "SpatialCrossMapLRNTest"
input : "data"
input_shape{dim:1 dim :3 dim:224 dim :224}
layer {
  name: "crossMapLrn"
  type: "LRN"
  bottom: "data"
  top: "crossMapLrn"
  lrn_param {
    local_size: 5
    alpha: 1.0E-4
    beta: 0.75
    k: 1.0
  }
}
"""
crossMapLrnShapes = [{"data": (1, 3, 224, 224)}]
crossMapLrnName = "crossMapLrn"


# Test Data for SpatialWithinChannelLRN
withinChannelLRNDefinition = """
name : "SpatialWithinChannelLRNTest"
input : "data"
input_shape{dim:1 dim :3 dim:224 dim :224}
layer {
  name: "withinChannelLRN"
  type: "LRN"
  bottom: "data"
  top: "withinChannelLRN"
  lrn_param {
    local_size: 5
    alpha: 1.0E-4
    beta: 0.75
    k: 1.0
    norm_region : WITHIN_CHANNEL
  }
}
"""
withinChannelLRNShapes = [{"data": (1, 3, 224, 224)}]
withinChannelLRNName = "withinChannelLRN"

# Test data for Inner product
innerProductDefinition = """
name : "InnerProductTest"
input : "data"
input_shape{dim: 2 dim: 10}
layer {
  name: "innerProduct"
  type: "InnerProduct"
  bottom: "data"
  top: "innerProduct"
  inner_product_param {
    num_output: 10
  }
}
"""

innerProductShapes = [{"data": (2, 10)}]
innerProductName = "innerProduct"

# Test data for max pooling
maxpoolingDefinition = """
name : "MaxpoolingTest"
input : "data"
input_shape{dim: 1 dim: 3 dim: 3 dim: 3}
layer {
  name: "maxpooling"
  type: "Pooling"
  bottom: "data"
  top: "maxpooling"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
"""
maxpoolingShapes = [{"data": (1, 3, 3, 3)}]
maxpoolingName = "maxpooling"

# Test data for average pooling
avepoolingDefinition = """
name : "AvepoolingTest"
input : "data"
input_shape{dim: 1 dim: 3 dim: 3 dim: 3}
layer {
  name: "avepooling"
  type: "Pooling"
  bottom: "data"
  top: "avepooling"
  pooling_param {
    pool: AVE
    kernel_size: 2
    stride: 2
  }
}
"""
avepoolingShapes = [{"data": (1, 3, 3, 3)}]
avepoolingName = "avepooling"

# Test data for SoftMax
softMaxDefinition = """
name : "SoftMaxTest"
input : "data"
input_shape{dim: 2 dim: 2}
layer {
  name: "softMax"
  type: "Softmax"
  bottom: "data"
  top: "softMax"
}
"""
softMaxShapes = [{"data": (2, 2)}]
softMaxName = "softMax"

# Test data for Tanh
tanhDefinition = """
name : "TanhTest"
input : "data"
input_shape{dim: 2 dim: 2}
layer {
  name: "tanh"
  type: "TanH"
  bottom: "data"
  top: "tanh"
}
"""
tanhShapes = [{"data": (2, 2)}]
tanhName = "tanh"

# Test data for Sigmoid
sigmoidDefinition = """
name : "SigmoidTest"
input : "data"
input_shape{dim: 2 dim: 2}
layer {
  name: "sigmoid"
  type: "Sigmoid"
  bottom: "data"
  top: "sigmoid"
}
"""
sigmoidShapes = [{"data": (2, 2)}]
sigmoidName = "sigmoid"

# Test data for Abs
absDefinition = """
name : "AbsTest"
input : "data"
input_shape{dim: 2 dim: 2}
layer {
  name: "abs"
  type: "AbsVal"
  bottom: "data"
  top: "abs"
}
"""
absShapes = [{"data": (2, 2)}]
absName = "abs"

# Test data for BatchNormalization
batchNormDefinition = """
name : "BatchNormTest"
input: "data"
input_dim: 1
input_dim: 3
input_dim: 224
input_dim: 224

layer {
        bottom: "data"
        top: "conv1"
        name: "conv1"
        type: "Convolution"
        convolution_param {
                num_output: 64
                kernel_size: 7
                pad: 3
                stride: 2
        }
}

layer {
        bottom: "conv1"
        top: "batchNorm"
        name: "batchNorm"
        type: "BatchNorm"
        batch_norm_param {
                use_global_stats: true
        }
}
"""
batchNormShapes = [{"data": (1, 3, 224, 224)}]
batchNormName = "batchNorm"

# Test data for Concat
concatDefinition = """
name : "ConcatTest"
input : "data1"
input_shape{dim: 2 dim: 2}
input : "data2"
input_shape{dim: 2 dim: 2}
layer {
  name: "abs"
  type: "AbsVal"
  bottom: "data1"
  top: "abs"
}
layer {
  name: "sigmoid"
  type: "Sigmoid"
  bottom: "data2"
  top: "sigmoid"
}
layer {
  name: "concat"
  type: "Concat"
  bottom: "abs"
  bottom: "sigmoid"
  top: "concat"
}
"""
concatShapes = [{"data1": (2, 2)}, {"data2": (2, 2)}]
concatName = "concat"

# Test data for Elu
eluDefinition = """
name : "EluTest"
input : "data"
input_shape{dim: 2 dim: 2}
layer {
  name: "elu"
  type: "ELU"
  bottom: "data"
  top: "elu"
}
"""
eluShapes = [{"data": (2, 2)}]
eluName = "elu"

# Test data for Flattern
flattenDefinition = """
name : "FlattenTest"
input : "data"
input_shape{dim: 2 dim: 2}
layer {
  name: "flatten"
  type: "Flatten"
  bottom: "data"
  top: "flatten"
}
"""
flattenShapes = [{"data": (2, 2)}]
flattenName = "flatten"

# Test data for Log
logDefinition = """
name : "LogTest"
input : "data"
input_shape{dim: 2 dim: 2}
layer {
  name: "log"
  type: "Log"
  bottom: "data"
  top: "log"
}
"""
logShapes = [{"data": (2, 2)}]
logName = "log"

# Test data for Power
powerDefinition = """
name : "PowerTest"
input : "data"
input_shape{dim: 2 dim: 2}
layer {
  name: "power"
  type: "Power"
  bottom: "data"
  top: "power"
}
"""
powerShapes = [{"data": (2, 2)}]
powerName = "power"

# Test data for PReLU
preluDefinition = """
name : "PReLUTest"
input : "data"
input_shape{dim: 2 dim: 5}
layer {
  name: "prelu"
  type: "PReLU"
  bottom: "data"
  top: "prelu"
}
"""
preluShapes = [{"data": (2, 5)}]
preluName = "prelu"

# Test data for Reshape
reshapeDefinition = """
name : "ReshapeTest"
input : "data"
input_shape{dim: 2 dim: 8}
layer {
  name: "reshape"
  type: "Reshape"
  bottom: "data"
  top: "reshape"
  reshape_param { shape { dim:  0 dim:  -1  dim:  4 } }
}
"""
reshapeShapes = [{"data": (2, 8)}]
reshapeName = "reshape"

# Test data for Scale
scaleDefinition = """
name : "ScaleTest"
input : "data"
input_shape{dim: 2 dim: 2}
layer {
  name: "scale"
  type: "Scale"
  bottom: "data"
  top: "scale"
}
"""
scaleShapes = [{"data": (2, 2)}]
scaleName = "scale"

# Test data for Bias
biasDefinition = """
name : "BiasTest"
input : "data"
input_shape{dim: 2 dim: 2}
layer {
  name: "bias"
  type: "Bias"
  bottom: "data"
  top: "bias"
}
"""
biasShapes = [{"data": (2, 2)}]
biasName = "bias"

# Test data for Threshold
thresholdDefinition = """
name : "ThresholdTest"
input : "data"
input_shape{dim: 2 dim: 2}
layer {
  name: "threshold"
  type: "Threshold"
  bottom: "data"
  top: "threshold"
  threshold_param {
    threshold : 0.5
  }
}
"""
thresholdShapes = [{"data": (2, 2)}]
thresholdName = "threshold"

# Test data for Exp
expDefinition = """
name : "ExpTest"
input : "data"
input_shape{dim: 2 dim: 2}
layer {
  name: "exp"
  type: "Exp"
  bottom: "data"
  top: "exp"
}
"""
expShapes = [{"data": (2, 2)}]
expName = "exp"

# Test data for Slice
sliceDefinition = """
name : "SliceTest"
input : "data"
input_shape{dim: 2 dim: 2}
layer {
  name: "slice"
  type: "Slice"
  bottom: "data"
  top: "slice"
}
"""
sliceShapes = [{"data": (2, 2)}]
sliceName = "slice"

# Test data for Tile
tileDefinition = """
name : "TileTest"
input : "data"
input_shape{dim: 2 dim : 2}
layer {
  name: "tile"
  type: "Tile"
  bottom: "data"
  top: "tile"
  tile_param {
    axis : 1
    tiles : 2
  }
}
"""
tileShapes = [{"data": (2, 2)}]
tileName = "tile"

# Test data for Eltwise MAX
eltwiseMaxDefinition = """
name : "EltwiseMaxTest"
input : "data1"
input_shape{dim: 2 dim: 2}
input : "data2"
input_shape{dim: 2 dim: 2}
layer {
  name: "abs"
  type: "AbsVal"
  bottom: "data1"
  top: "abs"
}
layer {
  name: "sigmoid"
  type: "Sigmoid"
  bottom: "data2"
  top: "sigmoid"
}
layer {
  name: "eltwiseMax"
  type: "Eltwise"
  bottom: "abs"
  bottom: "sigmoid"
  top: "eltwiseMax"
  eltwise_param {
    operation : MAX
  }
}
"""
eltwiseMaxShapes = [{"data1": (2, 2)}, {"data2": (2, 2)}]
eltwiseMaxName = "eltwiseMax"

# Test data for Eltwise Prod
eltwiseProdDefinition = """
name : "EltwiseProdTest"
input : "data1"
input_shape{dim: 2 dim: 2}
input : "data2"
input_shape{dim: 2 dim: 2}
layer {
  name: "abs"
  type: "AbsVal"
  bottom: "data1"
  top: "abs"
}
layer {
  name: "sigmoid"
  type: "Sigmoid"
  bottom: "data2"
  top: "sigmoid"
}
layer {
  name: "eltwiseProd"
  type: "Eltwise"
  bottom: "abs"
  bottom: "sigmoid"
  top: "eltwiseProd"
  eltwise_param {
    operation : PROD
  }
}
"""
eltwiseProdShapes = [{"data1": (2, 2)}, {"data2": (2, 2)}]
eltwiseProdName = "eltwiseProd"

# Test data for Eltwise SUM
eltwiseSUMDefinition = """
name : "EltwiseSUMTest"
input : "data1"
input_shape{dim: 2 dim: 2}
input : "data2"
input_shape{dim: 2 dim: 2}
layer {
  name: "abs1"
  type: "AbsVal"
  bottom: "data1"
  top: "abs1"
}
layer {
  name: "abs2"
  type: "AbsVal"
  bottom: "data2"
  top: "abs2"
}
layer {
  name: "eltwiseSUM"
  type: "Eltwise"
  bottom: "abs1"
  bottom: "abs2"
  top: "eltwiseSUM"
  eltwise_param {
    operation : SUM
     coeff: [0.5 , 1.0]
  }
}
"""
eltwiseSUMShapes = [{"data1": (2, 2)}, {"data2": (2, 2)}]
eltwiseSUMName = "eltwiseSUM"

deconvolutionDefinition = """
name : "deconvolution"
input : "data"
input_shape {dim:1 dim :3 dim :5 dim :5}
layer {
  name: "deconvolution"
  type: "Deconvolution"
  bottom: "data"
  top: "deconvolution"
  convolution_param {
    num_output: 4
    pad: 0
    kernel_size: 2
    stride: 2
    weight_filler {
      type: "xavier"
    }
  }
}

"""
deconvolutionShapes = [{"data": (1, 3, 5, 5)}]
deconvolutionName = "deconvolution"

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
registerTestLayer(reluName, reluDefinition, reluShapes)
registerTestLayer(crossMapLrnName, crossMapLrnDefinition, crossMapLrnShapes)
registerTestLayer(withinChannelLRNName, withinChannelLRNDefinition, withinChannelLRNShapes)
registerTestLayer(innerProductName, innerProductDefinition, innerProductShapes)
registerTestLayer(maxpoolingName, maxpoolingDefinition, maxpoolingShapes)
registerTestLayer(avepoolingName, avepoolingDefinition, avepoolingShapes)
registerTestLayer(softMaxName, softMaxDefinition, softMaxShapes)
registerTestLayer(tanhName, tanhDefinition, tanhShapes)
registerTestLayer(sigmoidName, sigmoidDefinition, sigmoidShapes)
registerTestLayer(absName, absDefinition, absShapes)
registerTestLayer(batchNormName, batchNormDefinition, batchNormShapes)
registerTestLayer(concatName, concatDefinition, concatShapes)
registerTestLayer(eluName, eluDefinition, eluShapes)
registerTestLayer(flattenName, flattenDefinition, flattenShapes)
registerTestLayer(logName, logDefinition, logShapes)
registerTestLayer(powerName, powerDefinition, powerShapes)
registerTestLayer(preluName, preluDefinition, preluShapes)
registerTestLayer(reshapeName, reshapeDefinition, reshapeShapes)
registerTestLayer(scaleName, scaleDefinition, scaleShapes)
registerTestLayer(biasName, biasDefinition, biasShapes)
registerTestLayer(thresholdName, thresholdDefinition, thresholdShapes)
registerTestLayer(expName, expDefinition, expShapes)
registerTestLayer(sliceName, sliceDefinition, sliceShapes)
registerTestLayer(tileName, tileDefinition, tileShapes)
registerTestLayer(eltwiseMaxName, eltwiseMaxDefinition, eltwiseMaxShapes)
registerTestLayer(eltwiseProdName, eltwiseProdDefinition, eltwiseProdShapes)
registerTestLayer(eltwiseSUMName, eltwiseSUMDefinition, eltwiseSUMShapes)
registerTestLayer(deconvolutionName, deconvolutionDefinition, deconvolutionShapes)
