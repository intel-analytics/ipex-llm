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

import tempfile
import numpy as np
import pytest
import tensorflow as tf

from bigdl.orca.test_zoo_utils import ZooTestCase
from bigdl.chronos.model.tf2.TCN_keras import model_creator, TemporalConvNet, TemporalBlock


def create_data():
    train_num_samples = 1000
    test_num_samples = 400
    past_seq_len = 20
    input_feature_num = 10
    output_feature_num = 2
    future_seq_len = 10

    def get_x_y(num_samples):
        x = np.random.randn(num_samples, past_seq_len, input_feature_num)
        y = np.random.randn(num_samples, future_seq_len, output_feature_num)
        return x, y
    train_data = get_x_y(train_num_samples)
    test_data = get_x_y(test_num_samples)
    return train_data, test_data


class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, schedule):
        super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule
    
    def on_epoch_begin(self, epochs, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        scheduled_lr = self.schedule(epochs, lr)
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)


LR_SCHEDULE = [(2, 1e-2), (4, 5e-3)]


@pytest.mark.skipif(tf.__version__ < '2.0.0', reason="Run only when tf>2.0.0.")
class TestTcnKeras(ZooTestCase):

    train_data, test_data = create_data()
    model = model_creator(config={
        "past_seq_len": 20,
        "future_seq_len": 10,
        "input_feature_num": 10,
        "output_feature_num": 2
    })

    def test_tcn_fit_predict_evaluate(self):
        self.model.fit(self.train_data[0],
                       self.train_data[1],
                       epochs=2,
                       validation_data=self.test_data)
        yhat = self.model.predict(self.test_data[0])
        self.model.evaluate(self.test_data[0], self.test_data[1])
        assert yhat.shape == self.test_data[1].shape

    def test_tcn_save_load(self):
        self.model.fit(self.train_data[0],
                       self.train_data[1],
                       epochs=2,
                       validation_data=self.test_data)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            self.model.save(tmp_dir_name)
            restore_model = tf.keras.models.load_model(tmp_dir_name,
                                                       custom_objects={"TemporalConvNet": TemporalConvNet,
                                                                       "TemporalBlock": TemporalBlock})
        model_res = self.model.evaluate(self.test_data[0], self.test_data[1])
        restore_model_res = restore_model.evaluate(self.test_data[0], self.test_data[1])
        np.testing.assert_almost_equal(model_res, restore_model_res, decimal=5)
        assert isinstance(restore_model, TemporalConvNet)

    def test_transfer_learning_fine_tuning(self):
        # transfer_learning
        base_layer = self.model.layers[-3]
        base_layer.trainable = False

        inputs = tf.keras.layers.Input(shape=(20, 10))
        x = base_layer(inputs)
        x = tf.keras.layers.Permute((2, 1))(x)
        x = tf.keras.layers.Dense(10)(x)
        outputs = tf.keras.layers.Permute((2, 1))(x)
        new_model = tf.keras.Model(inputs, outputs)
        new_model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(1e-1), metrics=["mse"])
        new_model.fit(x=np.random.randn(100, 20, 10),
                      y=np.random.randn(100, 10, 2),
                      epochs=2, batch_size=32)
        assert new_model.layers[1].trainable == False

        # fine_tuning
        new_model.layers[1].trainable = True
        new_model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(5e-3), metrics=["mse"])
        new_model.fit(x=np.random.randn(100, 20, 10),
                      y=np.random.randn(100, 10, 2),
                      epochs=2, batch_size=32)

    def test_custom_callback(self):
        model = model_creator(config={"input_feature_num": 10,
                                      "output_feature_num": 2,
                                      "past_seq_len": 20,
                                      "future_seq_len": 10,
                                      "lr": 1e-1})
        np.testing.assert_array_almost_equal(model.optimizer.lr.numpy(), np.asarray(1e-1))
        def lr_schedule(epoch, lr):
            if epoch<LR_SCHEDULE[0][0] or epoch>LR_SCHEDULE[-1][0]:
                return lr
            for i in range(len(LR_SCHEDULE)):
                if epoch==LR_SCHEDULE[i][0]:
                    return LR_SCHEDULE[i][1]
            return lr
        model.fit(self.train_data[0],
                  self.train_data[1],
                  batch_size=32,
                  epochs=5,
                  callbacks=[CustomLearningRateScheduler(lr_schedule)])
        np.testing.assert_array_almost_equal(model.optimizer.lr.numpy(), np.asarray(5e-3))


if __name__ == '__main__':
    pytest.main([__file__])
