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
import tensorflow as tf

import pytest

from bigdl.orca.test_zoo_utils import ZooTestCase
from bigdl.chronos.model.tf2.Seq2Seq_keras import LSTMSeq2Seq, model_creator
import numpy as np


def create_data():
    train_num_samples = 1000
    test_num_samples = 400
    input_feature_num = 10
    output_feature_num = 2
    past_seq_len = 10
    future_seq_len = np.random.randint(1, 5)

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
class TestSeq2Seq(ZooTestCase):

    train_data, test_data = create_data()
    model = model_creator(config={
        "input_feature_num": 10,
        "output_feature_num": 2,
        "future_seq_len": test_data[-1].shape[1],
        "lstm_hidden_dim": 32
    })

    def test_seq2seq_fit_predict_evaluate(self):
        self.model.fit(self.train_data[0],
                       self.train_data[1],
                       epochs=2,
                       validation_data=self.test_data)
        yhat = self.model.predict(self.test_data[0])
        self.model.evaluate(self.test_data[0], self.test_data[1])
        assert yhat.shape == (400, self.train_data[-1].shape[1], 2)

    def test_seq2seq_save_load(self):
        self.model.fit(self.train_data[0],
                       self.train_data[1],
                       epochs=2,
                       validation_data=self.test_data)
        with tempfile.TemporaryDirectory() as tmp_dir_file:
            self.model.save(tmp_dir_file)
            import keras
            restore_model = keras.models.load_model(tmp_dir_file,
                                                    custom_objects={"LSTMSeq2Seq": LSTMSeq2Seq})
        model_res = self.model.evaluate(self.test_data[0], self.test_data[1])
        restore_model_res = restore_model.evaluate(self.test_data[0], self.test_data[1])
        np.testing.assert_almost_equal(model_res, restore_model_res, decimal=5)
        assert isinstance(restore_model, LSTMSeq2Seq)

    def test_seq2seq_freeze_training(self):
        freeze_yhat = self.model(self.test_data[0], training=False)
        _freeze_yhat = self.model(self.test_data[0], training=False)
        assert np.all(_freeze_yhat == freeze_yhat)

        _unfreeze_yhat = self.model(self.test_data[0], training=True)
        unfreeze_yhat = self.model(self.test_data[0], training=True)
        assert np.any(_unfreeze_yhat != unfreeze_yhat)

    def test_transfer_learning_fine_tuning(self):
        # transfer_learning
        base_layer = self.model.layers[1: 3]
        for layer in base_layer:
            layer.trainable = False
        
        inputs = tf.keras.layers.Input(shape=(None, 10))
        states = base_layer[0](inputs)
        x = tf.keras.layers.Reshape((1, 2), input_shape=(2,))(inputs[:, -1, :2])
        x = base_layer[-1](x, states=states)
        outputs = tf.keras.layers.Dense(2)(x)
        new_model = tf.keras.Model(inputs, outputs)
        new_model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(1e-1), metrics=["mse"])
        new_model.fit(x=np.random.randn(100, 10, 10),
                      y=np.random.randn(100, self.test_data[-1].shape[1], 2),
                      epochs=2, batch_size=32)
        # encoder_layer
        assert new_model.layers[3].trainable == False

        # fine_tuning
        new_model.layers[4].trainable = True
        new_model.layers[5].trainable = True
        new_model.fit(x=np.random.randn(100, 10, 10),
                      y=np.random.randn(100, self.test_data[-1].shape[1], 2),
                      epochs=2, batch_size=32)

    def test_custom_callback(self):
        model = model_creator(config={"input_feature_num": 10,
                                      "output_feature_num": 2,
                                      "future_seq_len":self.test_data[-1].shape[1],
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
