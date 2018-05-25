# Analytics Zoo Anomaly detection

Analytics Zoo shows how to detect anomalies in time series data based on RNN network. Currently, a [python notebook](https://github.com/intel-analytics/analytics-zoo/tree/master/apps/anomaly-detection) is provided. 
In the example, a RNN network using Analytics Zoo Keras-Style API is built, and [NYC taxi passengers dataset](raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv) is used to train and test the model.

Build a RNN model for anomaly detection. There are three LSTM layers followed by one Dense layer at the end.
```python
model = Sequential()
model.add(LSTM(input_shape=(input_dim1, input_dim2, output_dim=8, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(15,return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(output_dim=1))
```
Train the model using Analytics Zoo APIs.
```python
model.compile(loss='mse', optimizer='rmsprop')
model.fit(x_train,y_train,batch_size=3028,nb_epoch=30)
```
Predict with given features.
```python
predictions = model.predict(x_test)
``` 
Anomalies could be defined by comparing the predictions and actual values. The current example defines data points as anomalies if the difference of predictions and actual values are larger than a certain value.
See more details in the exmaple [Python notebook](https://github.com/intel-analytics/analytics-zoo/tree/master/apps/anomaly-detection/anomaly-detection-nyc-taxi.ipynb)