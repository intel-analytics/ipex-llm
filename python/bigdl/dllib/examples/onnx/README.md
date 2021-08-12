# ONNX ResNet-50 Model Loading in BigDL


## Download model file
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

## How to run this example:
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
 
 
## Known issues:
  * ONNX feature only has Python API in BigDL.
  * Loaded ONNX model is limited for inference.
  * Most of operators defined in ONNX are not being supported by BigDL for now.
  * Missing feature of exporting BigDL model into ONNX format.
  