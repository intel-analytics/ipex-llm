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
* `dropouts`     Fraction of the input units to drop out. Float between 0 and 1.

## **Train an AnomalyDetector model**
After building the model, we can compile and train it using RDD of [Sample](https://bigdl-project.github.io/master/#APIGuide/Data/#sample).

Note that original features need to go through AnomalyDetector.unroll before being fed into the model. See more details in the examples.

**Scala**
```scala
import com.intel.analytics.bigdl.optim._

model.compile(optimizer = new RMSprop(learningRate = 0.001, decayRate = 0.9),
      loss = MeanSquaredError[Float]())
model.fit(trainRdd, batchSize = 1024, nbEpoch = 20)
```

**Python**
```python
model.compile(loss='mse', optimizer='rmsprop')
model.fit(train, batch_size = 1024, nb_epoch = 20)
```

---
## **Do prediction to detect anomalies**
After training the model, it can be used to predict values using previous data, then to detect anomalies.
Anomalies are defined by comparing the predictions and actual values. It ranks all the absolute difference of predictions and actual values with descending order, the top `anomalySize` data points are anomalies).

**Scala**
```scala
val yPredict = model.predict(testRdd).map(x => x.toTensor.toArray()(0))
val yTruth: RDD[Float] = testRdd.map(x => x.label.toArray()(0))
val anomalies = AnomalyDetector.detectAnomalies(yPredict, yTruth, 20)
```

**Python**
```python
y_predict = model.predict(test).map(lambda x: float(x[0]))
y_test = test.map(lambda x: float(x.label.to_ndarray()[0]))
anomalies = AnomalyDetector.detect_anomalies(y_test, y_predict, 20)
```

---
## **Examples**
We provide examples to train the AnomalyDetector model and detect possible anomalies using data of [NYC taxi passengers](https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv)

See [here](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/anomalydetection) for the Scala example.

See [here](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/examples/anomalydetection) for the Python example.

See a [Python notebook](https://github.com/intel-analytics/analytics-zoo/tree/master/apps/anomaly-detection) for defining and training a model using simple Keras layers, and more details. 