Analytics Zoo provides pre-defined models based on LSTM to detect anomalies in time series data. 
A sequence of values (e.g., last 50 hours) leading to the current time are used as input for the model, which then tries to predict the next data point. Anomalies are defined when actual values are distant from the model predictions.  

**Hightlights**

1. Keras style models, could use Keras style APIs(compile and fit), as well as NNFrames or BigDL Optimizer for training.
2. Models are defined base on LSTM.

---
## **Build an AnomalyDetction model**
You can call the following API in Scala and Python respectively to create an `AnomalyDetrctor` model

**Scala**
```scala
import com.intel.analytics.zoo.models.anomalydetection._
val model = AnomalyDetector(featureShape, hiddenLayers, dropouts)
```

* `featureShape` The input shape of features, fist dimension is unroll length, second dimension is feature size.
* `hiddenLayers` Units of hidden layers of LSTM.
* `dropouts`     Fraction of the input units to drop out. Float between 0 and 1.

**Python**
```
from zoo.models.anomalydetection import AnomalyDetector
model = AnomalyDetector(feature_shape=(10, 3), hidden_layers=[8, 32, 15], dropouts=[0.2, 0.2, 0.2])
```

* `feature_shape` The input shape of features, fist dimension is unroll length, second dimension is feature size.
* `hidden_layers` Units of hidden layers of LSTM.
* `dropouts`      Fraction of the input units to drop out. Float between 0 and 1.

## **Unroll features**
To prepare input for an AnomalyDetector model, you can use unroll a time series data with a unroll length.

**Scala**
```scala
val unrolled = AnomalyDetector.unroll(dataRdd, unrollLength, predictStep)
```

* `dataRdd`       RDD[Array]. data to be unrolled, it holds original time series features
* `unrollLength`  Int. the length of precious values to predict future value.
* `predictStep`   Int. How many time steps to predict future value, default is 1.

**Python**
```
unrolled = AnomalyDetector.unroll(data_rdd, unroll_length, predict_step)
```
* `data_rdd`       RDD[Array]. data to be unrolled, it holds original time series features
* `unroll_length`  Int. The length of precious values to predict future value.
* `predict_step`   Int. How many time steps to predict future value, default is 1.

---
## **Detect anomalies**
After training the model, it can be used to predict values using previous data, then to detect anomalies.
Anomalies are defined by comparing the predictions and actual values. It ranks all the absolute difference of predictions and actual values with descending order, the top `anomalySize` data points are anomalies).

**Scala**
```scala
val anomalies = AnomalyDetector.detectAnomalies(yTruth, yPredict, amonalySize)
```

* `yTruth`      RDD of float or double values. Truth to be compared. 
* `yPredict`    RDD of float or double values. Predictions.
* `anomalySize` Int. The size to be considered as anomalies.

**Python**``
```python
anomalies = AnomalyDetector.detect_anomalies(y_truth, y_predict, anomaly_size)
```

* `y_truth`      RDD of float or double values. Truth to be compared. 
* `y_predict`    RDD of float or double values. Predictions.
* `anomaly_size` Int. The size to be considered as anomalies.

---
## **Save Model**
After building and training an AnomalyDetector model, you can save it for future use.

**Scala**
```scala
model.saveModel(path, weightPath = null, overWrite = false)
```

* `path`: The path to save the model. Local file system, HDFS and Amazon S3 are supported. HDFS path should be like "hdfs://[host]:[port]/xxx". Amazon S3 path should be like "s3a://bucket/xxx".
* `weightPath`: The path to save weights. Default is null.
* `overWrite`: Whether to overwrite the file if it already exists. Default is false.

**Python**
```python
model.save_model(path, weight_path=None, over_write=False)
```

* `path`: The path to save the model. Local file system, HDFS and Amazon S3 are supported. HDFS path should be like 'hdfs://[host]:[port]/xxx'. Amazon S3 path should be like 's3a://bucket/xxx'.
* `weight_path`: The path to save weights. Default is None.
* `over_write`: Whether to overwrite the file if it already exists. Default is False.

---
## **Load Model**
To load an AnomalyDetector model (with weights) saved [above](#save-model):

**Scala**
```scala
AnomalyDetector.loadModel[Float](path, weightPath = null)
```

* `path`: The path for the pre-defined model. Local file system, HDFS and Amazon S3 are supported. HDFS path should be like "hdfs://[host]:[port]/xxx". Amazon S3 path should be like "s3a://bucket/xxx".
* `weightPath`: The path for pre-trained weights if any. Default is null.

**Python**
```python
AnomalyDetector.load_model(path, weight_path=None)
```

* `path`: The path for the pre-defined model. Local file system, HDFS and Amazon S3 are supported. HDFS path should be like 'hdfs://[host]:[port]/xxx'. Amazon S3 path should be like 's3a://bucket/xxx'.
* `weight_path`: The path for pre-trained weights if any. Default is None.
