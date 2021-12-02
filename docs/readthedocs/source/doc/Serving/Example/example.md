# Cluster Serving Example

There are some examples provided for new user or existing Tensorflow user.
## End-to-end Example
### TFDataSet: 
[l08c08_forecasting_with_lstm.py](https://github.com/intel-analytics/bigdl/tree/master/docs/docs/ClusterServingGuide/OtherFrameworkUsers/l08c08_forecasting_with_lstm.py)
### Tokenizer: 
[l10c03_nlp_constructing_text_generation_model.py](https://github.com/intel-analytics/bigdl/tree/master/docs/docs/ClusterServingGuide/OtherFrameworkUsers/l10c03_nlp_constructing_text_generation_model.py) 
### ImageDataGenerator: 
[transfer_learning.py](https://github.com/intel-analytics/bigdl/tree/master/docs/docs/ClusterServingGuide/OtherFrameworkUsers/transfer_learning.py)

## Model/Data Convert Guide
This guide is for users who:

* have written local code of Tensorflow, Pytorch(to be added)
* have used specified data type of a specific framework, e.g. TFDataSet
* want to deploy the local code on Cluster Serving but do not know how to write client code (Cluster Serving takes Numpy Ndarray as input, other types need to transform in advance).

**If you have the above needs but fail to find the solution below, please [create issue here](https://github.com/intel-analytics/bigdl/issues)

## Tensorflow

Model - includes savedModel, Frozen Graph (savedModel is recommended).

Data - includes [TFDataSet](#tfdataset), [Tokenizer](#tokenizer), [ImageDataGenerator](#imagedatagenerator)

Notes - includes tips to notice, includes [savedModel tips](#notes---use-savedmodel)

### Model - ckpt to savedModel
#### tensorflow all version
This method works in all version of TF

You need to create the graph, get the output layer, create place holder for input, load the ckpt then save the model
```
# --- code you need to write
input_layer = tf.placeholder(...)
model = YourModel(...)
output_layer = model.your_output_layer()
# --- code you need to write
with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_path))
    tf.saved_model.simple_save(sess,
                               FLAGS.export_path,
                               inputs={
                                   'input_layer': input_layer
                               },
                               outputs={"output_layer": output_layer})
```

#### tensorflow >= 1.15
This method works if you are familiar with savedModel signature, and tensorflow >= 1.15

model graph could be load via `.meta`, and load ckpt then save the model, signature_def_map is required to provide
```
# provide signature first
inputs = tf.placeholder(...)
outputs = tf.add(inputs, inputs)
tensor_info_input = tf.saved_model.utils.build_tensor_info(inputs)
tensor_info_output = tf.saved_model.utils.build_tensor_info(outputs)

prediction_signature = (
  tf.saved_model.signature_def_utils.build_signature_def(
      inputs={'x_input': tensor_info_input},
      outputs={'y_output': tensor_info_output},
      method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

      
# Your ckpt file is prefix.meta, prefix.index, etc
ckpt_prefix = 'model/model.ckpt-xxxx'
export_dir = 'saved_model'

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # load
    loader = tf.train.import_meta_graph(ckpt_prefix + '.meta')
    loader.restore(sess, ckpt_prefix)

    # export
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    builder.add_meta_graph_and_variables(sess,
                                         [tf.saved_model.tag_constants.TRAINING, tf.saved_model.tag_constants.SERVING],signature_def_map={
      tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
          prediction_signature 
      }
    )
    builder.save()
```
### Model - Keras to savedModel
#### tensorflow > 2.0
```
model = tf.keras.models.load_model("./model.h5")
tf.saved_model.save(model, "saved_model")
```
### Model - ckpt to Frozen Graph
[freeze checkpoint example](https://github.com/intel-analytics/bigdl/tree/master/pyzoo/bigdl/examples/tensorflow/freeze_checkpoint)
### Notes - Use SavedModel
If model has single tensor input, then nothing to notice.

**If model has multiple input, please notice following.**

When export, savedModel would store the inputs in alphabetical order. Use `saved_model_cli show --dir . --all` to see the order. e.g.
```
signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['id1'] tensor_info:
        dtype: DT_INT32
        shape: (-1, 512)
        name: id1:0
    inputs['id2'] tensor_info:
        dtype: DT_INT32
        shape: (-1, 512)
        name: id2:0

```

when enqueue to Cluster Serving, follow this order
### Data
To transform following data type to Numpy Ndarray, following examples are provided
