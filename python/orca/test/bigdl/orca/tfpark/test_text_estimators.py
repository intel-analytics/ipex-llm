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

import os
import random
from test.zoo.pipeline.utils.test_utils import ZooTestCase
from zoo.tfpark.text.estimator import *

resource_path = os.path.join(os.path.split(__file__)[0], "../resources")
bert_config_path = os.path.join(resource_path, "bert/bert_config.json")


class TestTextEstimators(ZooTestCase):

    def setup_method(self, method):
        tf.keras.backend.clear_session()
        super(TestTextEstimators, self).setup_method(method)

    def test_bert_classifier(self):
        def gen_record(has_label=True):
            res = dict()
            res["input_ids"] = np.random.randint(10000, size=2)
            res["input_mask"] = np.array([1] * 2)
            res["token_type_ids"] = np.array([0] * 1 + [1] * 1)
            if has_label:
                return res, np.array(random.choice([0, 1]))
            else:
                return res

        estimator = BERTClassifier(2, bert_config_path, optimizer=tf.train.AdamOptimizer())
        rdd = self.sc.parallelize([gen_record() for i in range(8)])
        # Training is too slow and memory consuming for a unit test. Skip here. Tested manually.
        # train_input_fn = bert_input_fn(rdd, 2, 4)
        # estimator.train(train_input_fn, 2)
        eval_input_fn = bert_input_fn(rdd, 2, 4)
        print(estimator.evaluate(eval_input_fn, eval_methods=["acc"]))
        test_rdd = self.sc.parallelize([gen_record(has_label=False) for i in range(4)])
        test_input_fn = bert_input_fn(test_rdd, 2, 4)
        predictions = estimator.predict(test_input_fn)
        assert predictions.count() == 4
        assert len(predictions.first()) == 2

    def test_bert_squad(self):
        def gen_record(has_label=True):
            res = dict()
            res["input_ids"] = np.random.randint(10000, size=2)
            res["input_mask"] = np.array([1] * 2)
            res["token_type_ids"] = np.array([0] * 1 + [1] * 1)
            if has_label:
                label = dict()
                label["start_position"] = np.array(0)
                label["end_position"] = np.array(0)
                return res, label
            else:
                res["unique_ids"] = np.array(np.random.randint(100))
                return res
        estimator = BERTSQuAD(bert_config_path, optimizer=tf.train.AdamOptimizer())
        # Training is too slow and memory consuming for a unit test. Skip here. Tested manually.
        # rdd = self.sc.parallelize([gen_record() for i in range(8)])
        # train_input_fn = bert_input_fn(rdd, 2, 4, labels={"start_positions", "end_positions"})
        # estimator.train(train_input_fn, 2)
        test_rdd = self.sc.parallelize([gen_record(has_label=False) for i in range(4)])
        test_input_fn = bert_input_fn(test_rdd, 2, 4, extra_features={"unique_ids": (tf.int32, [])})
        predictions = estimator.predict(test_input_fn)
        assert predictions.count() == 4
        assert isinstance(predictions.first(), dict)

    def test_bert_ner(self):
        def gen_record(has_label=True):
            res = dict()
            res["input_ids"] = np.random.randint(10000, size=2)
            res["input_mask"] = np.array([1] * 2)
            res["token_type_ids"] = np.array([0] * 1 + [1] * 1)
            if has_label:
                return res, np.array(np.random.randint(10, size=2))
            else:
                return res
        estimator = BERTNER(10, bert_config_path, optimizer=tf.train.AdamOptimizer())
        # Training is too slow and memory consuming for a unit test. Skip here. Tested manually.
        # rdd = self.sc.parallelize([gen_record() for i in range(8)])
        # train_input_fn = bert_input_fn(rdd, 2, 4, label_size=2)
        # estimator.train(train_input_fn, 2)
        test_rdd = self.sc.parallelize([gen_record(has_label=False) for i in range(4)])
        test_input_fn = bert_input_fn(test_rdd, 2, 4)
        predictions = estimator.predict(test_input_fn)
        assert predictions.count() == 4
        assert len(predictions.first()) == 2


if __name__ == "__main__":
    pytest.main([__file__])
