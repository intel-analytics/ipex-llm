---
## **Introduction**

Analytics Zoo Orca Tenorflow Estimator provides a set APIs for running TensorFlow model on Spark in a distributed fashion. 

__Remarks__:

- You need to install __tensorflow==1.15.0__ on your driver node.
- Your operating system (OS) is required to be one of the following 64-bit systems:
__Ubuntu 16.04 or later__ and __macOS 10.12.6 or later__.
- To run on other systems, you need to manually compile the TensorFlow source code. Instructions can
  be found [here](https://github.com/tensorflow/tensorflow/tree/v1.10.0/tensorflow/java).
---

### **Orca TF Estimator**

Orca TF Estimator is an estimator to do Tensorflow training/evaluation/prediction on Spark in a distributed fashion. 

It can support various data types, like XShards, Spark DataFrame, tf.data.Dataset, numpy.ndarrays, etc. 

It supports both native Tensorflow Graph model and Tensorflow Keras model in the unified APIs. 

For native Tensorflow Graph model, you can reference our [graph model example](https://github.com/intel-analytics/analytics-zoo/blob/master/docs/docs/colab-notebook/orca/quickstart/tf_lenet_mnist.ipynb).

For Tensorflow Keras model, you can reference our [keras model example](https://github.com/intel-analytics/analytics-zoo/blob/master/docs/docs/colab-notebook/orca/quickstart/keras_lenet_mnist.ipynb).

#### **Create Estimator with graph model**

You can creating Orca TF Estimator with native Tensorflow graph model.
```
from zoo.orca.learn.tf.estimator import Estimator

Estimator.from_graph(*, inputs, outputs=None, labels=None, loss=None, optimizer=None,
           clip_norm=None, clip_value=None,
           metrics=None, updates=None,
           sess=None, model_dir=None, backend="bigdl")
``` 
* `inputs`: input tensorflow tensors.
* `outputs`: output tensorflow tensors.
* `labels`: label tensorflow tensors.
* `loss`: The loss tensor of the TensorFlow model, should be a scalar
* `optimizer`: tensorflow optimization method.
* `clip_norm`: float >= 0. Gradients will be clipped when their L2 norm exceeds this value.
* `clip_value`:  a float >= 0 or a tuple of two floats.
        
  If `clip_value` is a float, gradients will be clipped when their absolute value exceeds this value.
  
  If `clip_value` is a tuple of two floats, gradients will be clipped when their value less than `clip_value[0]` or larger than `clip_value[1]`.
  
* `metrics`: dictionary of {metric_name: metric tensor}.
* `sess`: the current TensorFlow Session, if you want to used a pre-trained model, you should use the Session to load the pre-trained variables and pass it to estimator
* `model_dir`: location to save model checkpoint and summaries.
* `backend`: backend for estimator. Now it only can be "bigdl".

This method returns an Estimator object.

E.g. To create Orca TF Estimator with tf graph:
```
from zoo.orca.learn.tf.estimator import Estimator

class SimpleModel(object):
    def __init__(self):
        self.user = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.item = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.label = tf.placeholder(dtype=tf.int32, shape=(None,))

        feat = tf.stack([self.user, self.item], axis=1)
        self.logits = tf.layers.dense(tf.to_float(feat), 2)

        self.loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=self.logits,
                                                                          labels=self.label))

model = SimpleModel()

est = Estimator.from_graph(
            inputs=[model.user, model.item],
            labels=[model.label],
            outputs=[model.logits],
            loss=model.loss,
            optimizer=tf.train.AdamOptimizer(),
            metrics={"loss": model.loss})
```

#### **Train graph model with Estimator**
After an Estimator created, you can call estimator API to train Tensorflow graph model:
```
fit(data,
    epochs=1,
    batch_size=32,
    feature_cols=None,
    label_cols=None,
    validation_data=None,
    session_config=None,
    feed_dict=None,
    checkpoint_trigger=None
    )
```
* `data`: train data. It can be XShards, Spark DataFrame, tf.data.Dataset.
        
   If `data` is XShards, each element can be Pandas Dataframe or {'x': a feature numpy array
         or a tuple of feature numpy arrays, 'y': a label numpy array or a tuple of
         label numpy arrays}
   
   If `data` is tf.data.Dataset, each element is a tuple of input tensors.
   
* `epochs`: number of epochs to train.
* `batch_size`: total batch size for each iteration.
* `feature_cols`: feature column names if train data is Spark DataFrame.
* `label_cols`: label column names if train data is Spark DataFrame.
* `validation_data`: validation data. Validation data type should be the same as train data.
* `session_config`: tensorflow session configuration for training. Should be object of tf.ConfigProto
* `feed_dict`: a dictionary. The key is TensorFlow tensor, usually a placeholder, the value of the dictionary is a tuple of two elements. 
  
   The first one of the tuple is the value to feed to the tensor in training phase and the second one is the value to feed to the tensor in validation phase.
* `checkpoint_trigger`: when to trigger checkpoint during training. Should be bigdl optimzer trigger, like EveryEpoch(), SeveralIteration(num_iterations),etc.

Example of Train with Orca TF Estimator. 

1. Train data is tf.data.DataSet. E.g.
```
dataset = tf.data.Dataset.from_tensor_slices((np.random.randint(0, 200, size=(100,)),
                                              np.random.randint(0, 50, size=(100,)),
                                              np.ones(shape=(100,), dtype=np.int32)))
est.fit(data=dataset,
        batch_size=8,
        epochs=10,
        validation_data=dataset)
```

2.  Train data is Spark DataFrame. E.g.
```
est.fit(data=df,
        batch_size=8,
        epochs=10,
        feature_cols=['user', 'item'],
        label_cols=['label'],
        validation_data=df)
```

3.  Train data is [XShards](../data.md). E.g.
```
file_path = os.path.join(resource_path, "orca/learn/ncf.csv")
data_shard = zoo.orca.data.pandas.read_csv(file_path)

def transform(df):
    result = {
        "x": (df['user'].to_numpy(), df['item'].to_numpy()),
        "y": df['label'].to_numpy()
    }
    return result

data_shard = data_shard.transform_shard(transform)

est.fit(data=data_shard,
        batch_size=8,
        epochs=10,
        validation_data=data_shard)
```


#### **Create Estimator with Keras model**

You can creating Orca TF Estimator with Tensorflow Keras model. The model must be compiled.
```
from zoo.orca.learn.tf.estimator import Estimator

Estimator.from_keras(keras_model, metrics=None, model_dir=None, backend="bigdl")
```
* `keras_model`: the tensorflow.keras model, which must be compiled.
* `metrics`: user specified metric.
* `model_dir`: location to save model checkpoint and summaries.
* `backend`: backend for estimator. Now it only can be "bigdl".

This method returns an Estimator object.

E.g. To create Orca TF Estimator with tf keras model:
```
from zoo.orca.learn.tf.estimator import Estimator

user = tf.keras.layers.Input(shape=[1])
item = tf.keras.layers.Input(shape=[1])

feat = tf.keras.layers.concatenate([user, item], axis=1)
predictions = tf.keras.layers.Dense(2, activation='softmax')(feat)

model = tf.keras.models.Model(inputs=[user, item], outputs=predictions)
model.compile(optimizer='rmsprop',
              oss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
              
est = Estimator.from_keras(keras_model=model)              
```

#### **Train Keras model with Estimator**
After an Estimator created, you can call estimator's API to train Tensorflow Keras model:
```
fit(data,
    epochs=1,
    batch_size=32,
    feature_cols=None,
    label_cols=None,
    validation_data=None,
    session_config=None,
    checkpoint_trigger=None
    )
```
* `data`: train data. It can be XShards, Spark DataFrame, tf.data.Dataset.
        
   If `data` is XShards, each element needs to be {'x': a feature numpy array
         or a tuple of feature numpy arrays, 'y': a label numpy array or a tuple of
         label numpy arrays}
   
   If `data` is tf.data.Dataset, each element is a tuple of input tensors.
   
* `epochs`: number of epochs to train.
* `batch_size`: total batch size for each iteration.
* `feature_cols`: feature column names if train data is Spark DataFrame.
* `label_cols`: label column names if train data is Spark DataFrame.
* `validation_data`: validation data. Validation data type should be the same as train data.
* `session_config`: tensorflow session configuration for training. Should be object of tf.ConfigProto
* `checkpoint_trigger`: when to trigger checkpoint during training. Should be bigdl optimzer trigger, like EveryEpoch(), SeveralIteration(num_iterations),etc.

1. Train data is tf.data.DataSet. E.g.
```
dataset = tf.data.Dataset.from_tensor_slices((np.random.randint(0, 200, size=(100,)),
                                              np.random.randint(0, 50, size=(100,)),
                                              np.ones(shape=(100,), dtype=np.int32)))
                                              
dataset = dataset.map(lambda user, item, label: [(user, item), label])
                                             
est.fit(data=dataset,
        batch_size=8,
        epochs=10,
        validation_data=dataset)
```

2.  Train data is Spark DataFrame. E.g.
```
est.fit(data=df,
        batch_size=8,
        epochs=10,
        feature_cols=['user', 'item'],
        label_cols=['label'],
        validation_data=df)
```

3. If train data is XShards, e.g.
```
file_path = os.path.join(self.resource_path, "orca/learn/ncf.csv")
data_shard = zoo.orca.data.pandas.read_csv(file_path)

def transform(df):
    result = {
        "x": (df['user'].to_numpy().reshape([-1, 1]),
              df['item'].to_numpy().reshape([-1, 1])),
        "y": df['label'].to_numpy()
    }
    return result

data_shard = data_shard.transform_shard(transform)

est.fit(data=data_shard,
        batch_size=8,
        epochs=10,
        validation_data=data_shard)

```

#### **Evaluate with Estimator**

You can call estimator's API to evaluate Tensorflow graph model or keras model.
```
evaluate(data, batch_size=4,
         feature_cols=None,
         label_cols=None,
        )
```
* `data`: evaluation data. It can be XShards, Spark DataFrame, tf.data.Dataset.
        
   If `data` is XShards, each element needs to be {'x': a feature numpy array or a tuple of feature numpy arrays, 'y': a label numpy array or a tuple of label numpy arrays}
   
   If `data` is tf.data.Dataset, each element is `[feature tensor tuple, label tensor tuple]`
   
* `batch_size`: batch size per thread.
* `feature_cols`: feature_cols: feature column names if train data is Spark DataFrame.
* `label_cols`: label column names if train data is Spark DataFrame.

This method returns evaluation result as a dictionary in the format of {'metric name': metric value}

#### **Predict with Estimator**

You can call estimator's such APIs to predict with trained model.
```
predict(data, batch_size=4,
        feature_cols=None,
        ):
```
* `data`: data to be predicted. It can be XShards, Spark DataFrame, or tf.data.Dataset.        
        
    If `data` is XShard, each element needs to be {'x': a feature numpy array
         or a tuple of feature numpy arrays}.         
         
    If `data` is tf.data.Dataset, each element is feature tensor tuple.

* `batch_size`: batch size per thread
* `feature_cols`: list of feature column names if input data is Spark DataFrame.

This method returns a predicted result.

1. Predict data is tf.data.DataSet. The prediction result should be an XShards and each element is {'prediction': predicted numpy array or list of predicted numpy arrays}.
```
dataset = tf.data.Dataset.from_tensor_slices((np.random.randint(0, 200, size=(100, 1)),
                                              np.random.randint(0, 50, size=(100, 1))))
predictions = est.predict(dataset)

prediction_shards = est.predict(data_shard)
predictions = prediction_shards.collect()

assert 'prediction' in predictions[0]
```

2. Predict data is Spark DataFrame. The predict result is a DataFrame which includes original columns plus `prediction` column. The `prediction` column can be FloatType, VectorUDT or Array of VectorUDT depending on model outputs shape.
```
prediction_df = est.predict(df, batch_size=4, feature_cols=['user', 'item'])

assert 'prediction' in prediction_df.columns
```

3. Predict data is XShards. The prediction result should be an XShards and each element is {'prediction': predicted numpy array or list of predicted numpy arrays}.
```
file_path = os.path.join(resource_path, "orca/learn/ncf.csv")
data_shard = zoo.orca.data.pandas.read_csv(file_path)

def transform(df):
    result = {
        "x": (df['user'].to_numpy(), df['item'].to_numpy())
    }
    return result

data_shard = data_shard.transform_shard(transform)

prediction_shards = est.predict(data_shard)
predictions = prediction_shards.collect()

assert 'prediction' in predictions[0]
```

#### **Checkpointing and Resume Training**

During training, Orca TF Estimator would save Orca checkpoint every epoch. You can also specify `checkpoint_trigger` in fit() to set checkpoint interval. The Orca checckpoints are saved in `model_dir` which is specified when you create estimator. You can load previous Orca checkpoint and resume train with it with such APIs:
 
```
load_orca_checkpoint(path, version=None)
``` 
* path: directory containing Orca checkpoint files.

With `version=None`, this method load latest checkpoint under specified directory.

E.g.
```
est = Estimator.from_keras(keras_model=model, model_dir=model_dir)
est.load_orca_checkpoint(model_dir, version=None)
est.fit(data=data_shard,
        batch_size=8,
        epochs=10,
        validation_data=data_shard,
        checkpoint_trigger=SeveralIteration(4))
```

If you want to load specified version of checkpoint, you can use:

```
load_orca_checkpoint(path, version)
```
* path: checkpoint directory which contains model.* and optimMethod-TFParkTraining.* files.
* version: checkpoint version, which is the suffix of model.* file, i.e., for modle.4 file, the version is 4.

After loading checkpoint, you can resume training with fit(). 

#### **Tensorboard support**

During training and validation, Orca TF Estimator would save Tensorflow summary data under `model_dir` which is specified when you create estimator. This data can be visualized in TensorBoard, or you can use estimator's APIs to retrieve it. If you want to save train/validation summary data to different directory or you don't specify `model_dir`, you can set the logdir and app name with such API:
```
set_tensorboard(log_dir, app_name)
```
* `log_dir`: The base directory path to store training and validation logs.
* `app_name`: The name of the application.

This method sets summary information during the training process for visualization purposes. Saved summary can be viewed via TensorBoard. In order to take effect, it needs to be called before fit.

Training summary will be saved to 'log_dir/app_name/train' and validation summary (if any) will be saved to 'log_dir/app_name/validation'.

E.g. Set tensorboard for the estimator:
```
est = Estimator.from_keras(keras_model=model)
log_dir = os.path.join(temp, "log")
est.set_tensorboard(log_dir, "test")
```

Now, you can see the summary data in TensorBoard. Or else, you can get summary data with such APIs:
```
get_train_summary(tag)
```
* `tag`: The string variable represents the scalar wanted. It can only be "Loss", "LearningRate", or "Throughput".

This method get the scalar from model train summary. Return list of summary data of [iteration_number, scalar_value, timestamp].

E.g.
```
est.fit(data=data_shard,
        batch_size=8,
        epochs=10,
        validation_data=data_shard)

assert os.path.exists(os.path.join(log_dir, "test/train"))
assert os.path.exists(os.path.join(log_dir, "test/validation"))

train_loss = est.get_train_summary("Loss")
```

```
get_validation_summary(tag)
```
* `tag`: The string variable represents the scalar wanted.

This method gets the scalar from model validation summary. Return list of summary data of [iteration_number, scalar_value, timestamp]

E.g.
```
val_scores = est.get_validation_summary("Loss")
```

#### **Save model**
After training, you can save model in the estimator:

##### **Save Tensorflow checkpoint**
```
save_tf_checkpoint(path)
```
* `path`: tensorflow checkpoint path.

If you use tensorflow graph model in this estimator, this method would save tensorflow checkpoint.

E.g.
```
temp = tempfile.mkdtemp()
model_checkpoint = os.path.join(temp, 'test.ckpt')
est.save_tf_checkpoint(model_checkpoint)
```

##### **Save TF Keras model**
```
save_keras_model(path, overwrite=True)
```
* `path`: keras model save path.
* `overwrite`: Boolean. Whether to silently overwrite any existing file at the target location. Default: True.

If you use tensorflow keras model in this estimator, this method would save keras model in specified path.

E.g.
```
temp = tempfile.mkdtemp()
model_path = os.path.join(temp, 'test.h5')
est.save_keras_model(model_path, overwrite=True)
```

