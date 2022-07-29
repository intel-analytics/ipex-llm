# Python DLLib Getting Start Guide

## 1. Code initialization
```nncontext``` is the main entry for provisioning the dllib program on the underlying cluster (such as K8s or Hadoop cluster), or just on a single laptop.

It is recommended to initialize `nncontext` at the beginning of your program:
```
from bigdl.dllib.nncontext import *
sc = init_nncontext()
```
For more information about ```nncontext```, please refer to [nncontext](https://bigdl.readthedocs.io/en/latest/doc/DLlib/Overview/dllib.html#nn-context)

## 3. Distributed Data Loading

#### Using Spark Dataframe APIs
DLlib supports Spark Dataframes as the input to the distributed training, and as
the input/output of the distributed inference. Consequently, the user can easily
process large-scale dataset using Apache Spark, and directly apply AI models on
the distributed (and possibly in-memory) Dataframes without data conversion or serialization

We create Spark session so we can use Spark API to load and process the data
```
spark = SQLContext(sc)
```

1. We can use Spark API to load the data into Spark DataFrame, eg. read csv file into Spark DataFrame
```
path = "pima-indians-diabetes.data.csv"
spark.read.csv(path)
```

If the feature column for the model is a Spark ML Vector. Please assemble related columns into a Vector and pass it to the model. eg.
```
from pyspark.ml.feature import VectorAssembler
vecAssembler = VectorAssembler(outputCol="features")
vecAssembler.setInputCols(["num_times_pregrant", "plasma_glucose", "blood_pressure", "skin_fold_thickness", "2-hour_insulin", "body_mass_index", "diabetes_pedigree_function", "age"])
assemble_df = vecAssembler.transform(df)
assemble_df.withColumn("label", col("class").cast(DoubleType) + lit(1))
```

2. If the training data is image, we can use DLLib api to load image into Spark DataFrame. Eg.
```
imgPath = "cats_dogs/"
imageDF = NNImageReader.readImages(imgPath, sc)
```

It will load the images and generate feature tensors automatically. Also we need generate labels ourselves. eg:
```
labelDF = imageDF.withColumn("name", getName(col("image"))) \
        .withColumn("label", getLabel(col('name')))
```

Then split the Spark DataFrame into traing part and validation part
```
(trainingDF, validationDF) = labelDF.randomSplit([0.9, 0.1])
```

## 4. Model Definition

#### Using Keras-like APIs

To define a model, you can use the [Keras Style API](https://bigdl.readthedocs.io/en/latest/doc/DLlib/Overview/keras-api.html).
```
x1 = Input(shape=[8])
dense1 = Dense(12, activation="relu")(x1)
dense2 = Dense(8, activation="relu")(dense1)
dense3 = Dense(2)(dense2)
dmodel = Model(input=x1, output=dense3)
```

After creating the model, you will have to decide which loss function to use in training.

Now you can use `compile` function of the model to set the loss function, optimization method.
```
dmodel.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy")
```

Now the model is built and ready to train.

## 5. Distributed Model Training
Now you can use 'fit' begin the training, please set the label columns. Model Evaluation can be performed periodically during a training.
1. If the dataframe is generated using Spark apis, you also need set the feature columns. eg.
```
model.fit(df, feature_cols=["features"], label_cols=["label"], batch_size=4, nb_epoch=1)
```
Note: Above model accepts single input(column `features`) and single output(column `label`).

If your model accepts multiple inputs(eg. column `f1`, `f2`, `f3`), please set the features as below:
```
model.fit(df, feature_cols=["f1", "f2"], label_cols=["label"], batch_size=4, nb_epoch=1)
```

Similarly, if the model accepts multiple outputs(eg. column `label1`, `label2`), please set the label columns as below:
```
model.fit(df, feature_cols=["features"], label_cols=["l1", "l2"], batch_size=4, nb_epoch=1)
```

2. If the dataframe is generated using DLLib `NNImageReader`, we don't need set `feature_cols`, we can set `transform` to config how to process the images before training. Eg.
```
from bigdl.dllib.feature.image import transforms
transformers = transforms.Compose([ImageResize(50, 50), ImageMirror()])
model.fit(image_df, label_cols=["label"], batch_size=1, nb_epoch=1, transform=transformers)
```
For more details about how to use DLLib keras api to train image data, you may want to refer [ImageClassification](https://github.com/intel-analytics/BigDL/tree/main/python/dllib/examples/keras/image_classification.py)

## 6. Model saving and loading
When training is finished, you may need to save the final model for later use.

BigDL allows you to save your BigDL model on local filesystem, HDFS, or Amazon s3.
- **save**
```
modelPath = "/tmp/demo/keras.model"
dmodel.saveModel(modelPath)
```

- **load**
```
loadModel = Model.loadModel(modelPath)
preDF = loadModel.predict(df, feature_cols=["features"], prediction_col="predict")
```

You may want to refer [Save/Load](https://bigdl.readthedocs.io/en/latest/doc/DLlib/Overview/keras-api.html#save)

## 7. Distributed evaluation and inference
After training finishes, you can then use the trained model for prediction or evaluation.

- **inference**
1. For dataframe generated by Spark API, please set `feature_cols` and `prediction_col`
```
dmodel.predict(df, feature_cols=["features"], prediction_col="predict")
```
2. For dataframe generated by `NNImageReader`, please set `prediction_col` and you can set `transform` if needed
```
model.predict(df, prediction_col="predict", transform=transformers)
```

- **evaluation**
Similary for dataframe generated by Spark API, the code is as below:
```
dmodel.evaluate(df, batch_size=4, feature_cols=["features"], label_cols=["label"])
```

For dataframe generated by `NNImageReader`:
```
model.evaluate(image_df, batch_size=1, label_cols=["label"], transform=transformers)
```

## 8. Checkpointing and resuming training
You can configure periodically taking snapshots of the model.
```
cpPath = "/tmp/demo/cp"
dmodel.set_checkpoint(cpPath)
```
You can also set ```over_write``` to ```true``` to enable overwriting any existing snapshot files

After training stops, you can resume from any saved point. Choose one of the model snapshots to resume (saved in checkpoint path, details see Checkpointing). Use Models.loadModel to load the model snapshot into an model object.
```
loadModel = Model.loadModel(path)
```

## 9. Monitor your training

- **Tensorboard**

BigDL provides a convenient way to monitor/visualize your training progress. It writes the statistics collected during training/validation. Saved summary can be viewed via TensorBoard.

In order to take effect, it needs to be called before fit.
```
dmodel.set_tensorboard("./", "dllib_demo")
```
For more details, please refer [visulization](visualization.md)

## 10. Transfer learning and finetuning

- **freeze and trainable**
BigDL DLLib supports exclude some layers of model from training.
```
dmodel.freeze(layer_names)
```
Layers that match the given names will be freezed. If a layer is freezed, its parameters(weight/bias, if exists) are not changed in training process.

BigDL DLLib also support unFreeze operations. The parameters for the layers that match the given names will be trained(updated) in training process
```
dmodel.unFreeze(layer_names)
```
For more information, you may refer [freeze](freeze.md)

## 11. Hyperparameter tuning
- **optimizer**

DLLib supports a list of optimization methods.
For more details, please refer [optimization](optim-Methods.md)

- **learning rate scheduler**

DLLib supports a list of learning rate scheduler.
For more details, please refer [lr_scheduler](learningrate-Scheduler.md)

- **batch size**

DLLib supports set batch size during training and prediction. We can adjust the batch size to tune the model's accuracy.

- **regularizer**

DLLib supports a list of regularizers.
For more details, please refer [regularizer](regularizers.md)

- **clipping**

DLLib supports gradient clipping operations.
For more details, please refer [gradient_clip](clipping.md)

## 12. Running program
```
python you_app_code.py
```
