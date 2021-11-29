## Introduction

This example demonstrates how to transform a TensorFlow saved model into a frozen model that can be used by TFNet.

### Usage:

Transform a saved model to a frozen model.

```bash
python freeze.py --saved_model_path /path/to/saved_model --output_path /path/to/tfnet
```

Use the frozen model in TFNet.

```python
from zoo.tfpark import TFNet
net = TFNet.from_export_folder("/path/to/tfnet")
```

