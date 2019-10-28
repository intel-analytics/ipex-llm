#ONNX (Open Neural Network Exchange)
[ONNX](https://onnx.ai/) is an open format to represent deep learning models.
As an active member of the deep learning community,BigDL continues to enrich itself
with supporting models defined in ONNX format, therefore allows our users to take 
advantages of the functionality BigDL offers without worrying about the cost of switching
among different model formats defined by various frameworks, as long as their models are
able to interpreted by ONNX.

## How to load ONNX model:
  * [Download model](https://s3.amazonaws.com/download.onnx/models/opset_9/resnet50.tar.gz)
  * Uncompress the file
     ```
         tar -zxvf resnet50.tar.gz
         .
         ├── model.onnx
         ├── test_data_set_0
         ├── test_data_set_1
         └── ....
     ```
 * Import library dependencies
```
import numpy as np
from bigdl.contrib.onnx import load
```

 * Set target ONNX ResNet-50 model path
 ```
 restnet_path = "uncompressed/file/path/model.onnx"
 ```
   
 * Load ONNX ResNet-50 model into BigDL
 ```
 restnet = load(restnet_path)
 ```
 
 * Create a sample tensor and pass it through loaded BigDL model
 ```
 restnet_tensor = np.random.random([10, 3, 224, 224])   
 restnet_out = restnet.forward(restnet_tensor)
 ```
 
## Supported ONNX Operators
  * [AveragePool](https://github.com/onnx/onnx/blob/master/docs/Operators.md#AveragePool)
  * [BatchNormalization](https://github.com/onnx/onnx/blob/master/docs/Operators.md#BatchNormalization)
  * [Concat](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Concat)
  * [Constant](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Constant)
  * [Conv](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv)
  * [Gather](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gather)
  * [Gemm](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gemm)
  * [MaxPool](https://github.com/onnx/onnx/blob/master/docs/Operators.md#MaxPool)
  * [Relu](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Relu)
  * [Reshape](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Reshape)
  * [Shape](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Shape)
  * [Softmax](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Softmax)
  * [Sum](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sum)
  * [Unsqueeze](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Unsqueeze)
  
  
## Known issues:
  * ONNX feature only has Python API in BigDL.
  * Loaded ONNX model is limited for inference.
  * Most of operators defined in ONNX are not being supported by BigDL for now.
  * Missing feature of exporting BigDL model into ONNX format.
