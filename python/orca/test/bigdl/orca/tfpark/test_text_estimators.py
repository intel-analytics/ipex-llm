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
from test.zoo.pipeline.utils.test_utils import ZooTestCase
from zoo.tfpark.text.estimator import *

resource_path = os.path.join(os.path.split(__file__)[0], "../resources")
bert_config_path = os.path.join(resource_path, "bert/bert_config.json")


def gen_record(has_label=True):
    res = dict()
    res["input_ids"] = np.random.randint(10000, size=2)
    res["input_mask"] = np.array([1]*2)
    res["token_type_ids"] = np.array([0]*1+[1]*1)
    if has_label:
        import random
        return res, np.array(random.choice([0, 1]))
    else:
        return res


class TestTextEstimators(ZooTestCase):

    def test_bert_classifier(self):
        estimator = BERTClassifier(2, bert_config_path, optimizer=tf.train.AdamOptimizer())
        rdd = self.sc.parallelize([gen_record() for i in range(8)])
        # Training is a bit too slow for a unit test. Skip here.
        # train_input_fn = bert_input_fn(rdd, 128, 4)
        # estimator.train(train_input_fn, 2)
        eval_input_fn = bert_input_fn(rdd, 2, 4)
        print(estimator.evaluate(eval_input_fn, eval_methods=["acc"]))
        test_rdd = self.sc.parallelize([gen_record(has_label=False) for i in range(4)])
        test_input_fn = bert_input_fn(test_rdd, 2, 4)
        predictions = estimator.predict(test_input_fn)
        assert predictions.count() == 4


if __name__ == "__main__":
    pytest.main([__file__])
