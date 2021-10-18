## Introduction

This example demonstrates how to transform a TensorFlow checkpoint into a frozen model that can be used by TFNet.

### Usage:

Transform a checkpoint to a frozen model.

```bash
python freeze_checkpoint.py \
    --pbPath /path/to/pb_file \
    --ckptPath /path/to/checkpoint_file \
    --inputsName tensor_names_of_model_inputs \
    --outputsName tensor_names_of_model_outputs \
    -o /path/to/tfnet
```

Use the frozen model in TFNet.

```python
from zoo.tfpark import TFNet
net = TFNet.from_export_folder("/path/to/tfnet")
```