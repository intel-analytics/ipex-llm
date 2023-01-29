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
import pytest
from unittest import TestCase
import numpy as np
from ... import op_tf2
from bigdl.chronos.utils import LazyImport
tf = LazyImport('tensorflow')
LSTMSeq2Seq = LazyImport('bigdl.chronos.model.tf2.Seq2Seq_keras.LSTMSeq2Seq')
model_creator = LazyImport('bigdl.chronos.model.tf2.Seq2Seq_keras.model_creator')


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


@op_tf2
class TestSeq2Seq(TestCase): 

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_seq2seq_fit_predict_evaluate(self):
        train_data, test_data = create_data()
        model = model_creator(config={
            "input_feature_num": 10,
            "output_feature_num": 2,
            "future_seq_len": test_data[-1].shape[1],
            "lstm_hidden_dim": 32
        })

        model.fit(train_data[0],
                  train_data[1],
                  epochs=2,
                  validation_data=test_data)
        yhat = model.predict(test_data[0])
        model.evaluate(test_data[0], test_data[1])
        assert yhat.shape == (400, train_data[-1].shape[1], 2)

    def test_seq2seq_save_load(self):
        train_data, test_data = create_data()
        model = model_creator(config={
            "input_feature_num": 10,
            "output_feature_num": 2,
            "future_seq_len": test_data[-1].shape[1],
            "lstm_hidden_dim": 32
        })
        model.fit(train_data[0],
                  train_data[1],
                  epochs=2,
                  validation_data=test_data)
        with tempfile.TemporaryDirectory() as tmp_dir_file:
            model.save(tmp_dir_file)
            import keras
            restore_model = keras.models.load_model(tmp_dir_file,
                                                    custom_objects={"LSTMSeq2Seq": LSTMSeq2Seq})
        model_res = model.evaluate(test_data[0], test_data[1])
        restore_model_res = restore_model.evaluate(test_data[0], test_data[1])
        np.testing.assert_almost_equal(model_res, restore_model_res, decimal=5)
        temp_LSTMSeq2Seq = LSTMSeq2Seq(input_feature_num=10,
                                       output_feature_num=2,
                                       future_seq_len=test_data[-1].shape[1],
                                       lstm_hidden_dim=32).__class__
        assert isinstance(restore_model, temp_LSTMSeq2Seq)

    def test_seq2seq_freeze_training(self):
        train_data, test_data = create_data()
        model = model_creator(config={
            "input_feature_num": 10,
            "output_feature_num": 2,
            "future_seq_len": test_data[-1].shape[1],
            "lstm_hidden_dim": 32
        })
        freeze_yhat = model(test_data[0], training=False)
        _freeze_yhat = model(test_data[0], training=False)
        assert np.all(_freeze_yhat == freeze_yhat)

        _unfreeze_yhat = model(test_data[0], training=True)
        unfreeze_yhat = model(test_data[0], training=True)
        assert np.any(_unfreeze_yhat != unfreeze_yhat)

if __name__ == '__main__':
    pytest.main([__file__])
