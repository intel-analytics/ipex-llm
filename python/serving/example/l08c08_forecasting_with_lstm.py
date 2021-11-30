#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Related url: https://github.com/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l08c08_forecasting_with_lstm.ipynb
# Forecasting with LSTM
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

# Get the trend with time and slope
def trend(time, slope=0):
    return slope * time


# Get a specific pattern, which can be customerized
def seasonal_pattern(season_time):
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

# Repeats the same pattern at each period
def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

# Obtain a random white noise
def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

# Convert the series to dataset form
def ndarray_to_dataset(ndarray):
    return tf.data.Dataset.from_tensor_slices(ndarray)

# Convert the series to dataset with some modifications
def sequential_window_dataset(series, window_size):
    series = tf.expand_dims(series, axis=-1)
    ds = ndarray_to_dataset(series)
    ds = ds.window(window_size + 1, shift=window_size, drop_remainder=True)
    ds = ds.flat_map(lambda window: window.batch(window_size + 1))
    ds = ds.map(lambda window: (window[:-1], window[1:]))
    return ds.batch(1).prefetch(1)

# Convert dataset form to ndarray
def dataset_to_ndarray(dataset):
    array=list(dataset.as_numpy_iterator())
    return np.ndarray(array)

# Generate some raw test data
time_range=4 * 365 + 1
time = np.arange(time_range)

slope = 0.05
baseline = 10
amplitude = 40
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)

noise_level = 5
noise = white_noise(time, noise_level, seed=42)

series += noise

# Modify the raw test data with DataSet form
tf.random.set_seed(42)
np.random.seed(42)

window_size = 30
test_set = sequential_window_dataset(series, window_size)

# Convert the DataSet form data to ndarry
#pre_in=series[np.newaxis, :, np.newaxis]
test_array=dataset_to_ndarray(test_set)

# Load the saved LSTM model
model=tf.keras.models.load_model("path/to/model")

# Predict with LSTM model
rnn_forecast_nd = model.predict(test_array)
