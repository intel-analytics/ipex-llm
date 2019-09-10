#
# Copyright 2018 Analytics Zoo Authors.
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

import pytest

import numpy as np
from test.zoo.pipeline.utils.test_utils import ZooTestCase
from zoo.tfpark.text.keras import *


class TestTextModels(ZooTestCase):

    def setup_method(self, method):
        import tensorflow as tf
        tf.keras.backend.clear_session()
        super(TestTextModels, self).setup_method(method)

    def test_intent_entity(self):
        model = IntentEntity(num_intents=8, num_entities=5, word_length=10,
                             word_vocab_size=200, char_vocab_size=50)
        input_data = [np.random.randint(200, size=(8, 30)), np.random.randint(50, size=(8, 30, 10))]
        output = model.predict(input_data, distributed=True)
        assert isinstance(output, list) and len(output) == 2
        assert output[0].shape == (8, 8)
        assert output[1].shape == (8, 30, 5)
        self.assert_tfpark_model_save_load(model, input_data)

    def test_ner_crf_reg_mode(self):
        model = NER(num_entities=10, word_length=5, word_vocab_size=20, char_vocab_size=10)
        input_data = [np.random.randint(20, size=(15, 12)), np.random.randint(10, size=(15, 12, 5))]
        output = model.predict(input_data, distributed=True)
        assert output.shape == (15, 12, 10)
        self.assert_tfpark_model_save_load(model, input_data)

    def test_ner_crf_pad_mode(self):
        model = NER(num_entities=15, word_length=8, word_vocab_size=20,
                    char_vocab_size=10, crf_mode="pad")
        input_data = [np.random.randint(20, size=(4, 12)), np.random.randint(10, size=(4, 12, 8)),
                      np.random.randint(12, size=(15, 1))]
        output = model.predict(input_data, distributed=True)
        assert output.shape == (4, 12, 15)
        self.assert_tfpark_model_save_load(model, input_data)

    def test_sequence_tagger_softmax(self):
        model = SequenceTagger(num_pos_labels=5, num_chunk_labels=10, word_vocab_size=150)
        input_data = np.random.randint(150, size=(10, 50))
        output = model.predict(input_data, distributed=True)
        assert isinstance(output, list) and len(output) == 2
        assert output[0].shape == (10, 50, 5)
        assert output[1].shape == (10, 50, 10)
        self.assert_tfpark_model_save_load(model, input_data)

    def test_sequence_tagger_crf(self):
        model = SequenceTagger(num_pos_labels=8, num_chunk_labels=8, word_vocab_size=50,
                               char_vocab_size=20, classifier="crf")
        input_data = [np.random.randint(50, size=(10, 15)),
                      np.random.randint(20, size=(10, 15, 12))]
        output = model.predict(input_data, distributed=True)
        assert isinstance(output, list) and len(output) == 2
        assert output[0].shape == (10, 15, 8)
        assert output[1].shape == (10, 15, 8)
        self.assert_tfpark_model_save_load(model, input_data)


if __name__ == "__main__":
    pytest.main([__file__])
