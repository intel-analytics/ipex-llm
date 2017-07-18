# Tensorflow model support example
BigDL support read/save tensorflow model. Here's an example how to use this feature.

Before you run this example, you need to install tensorflow on your machine. This can be simply done
by

```bash
pip install tensorflow
```

## Load tensorflow model
1. Generate tensorflow model
```bash
python model.py
```

2. Freeze tensorflow model
```bash
wget https://raw.githubusercontent.com/tensorflow/tensorflow/v1.0.0/tensorflow/python/tools/freeze_graph.py
python freeze_graph.py --input_graph model/model.pbtxt --input_checkpoint model/model.chkp --output_node_names="LeNet/fc4/BiasAdd" --output_graph "model.pb"
```

3. Run BigDL
```bash

```