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
wget https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py
python freeze_graph.py --input_graph model/model.pbtxt --input_checkpoint model/model.chkp --output_node_names="LeNet/fc4/BiasAdd" --output_graph "model.pb"
```

3. Run BigDL
```bash
spark-submit --master local[1] --class com.intel.analytics.bigdl.example.tensorflow.loadandsave.Load BigDL_jar_file ./model.pb
```

## Save BigDL model as tensorflow model
1. Run BigDL
```bash
spark-submit --master local[1] --class com.intel.analytics.bigdl.example.tensorflow.loadandsave.Save BigDL_jar_file
```

2. Generate summary file, you can find the dump_tf_graph.py in the bin folder of the dist package, or script folder of
the code
```bash
python dump_tf_graph.py model.pb
```

3. See the saved model via tensorboard
```bash
tensorboard --logdir ./log
```
