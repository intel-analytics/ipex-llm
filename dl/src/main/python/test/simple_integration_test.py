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
# Still in experimental stage!

from nn.layer import *
from nn.criterion import *
from optim.optimizer import *
from util.common import *
import numpy as np
import unittest
import tempfile


class TestWorkFlow(unittest.TestCase):
    def setUp(self):
        sparkConf = create_spark_conf()
        self.sc = SparkContext(master="local[4]", appName="test model",
                               conf=sparkConf)
        init_engine()

    def tearDown(self):
        self.sc.stop()

    def test_set_seed(self):
        l1 = Linear(10, 20, "Xavier").set_name("linear1").set_seed(1234).reset()  # noqa
        l2 = Linear(10, 20, "Xavier").set_name("linear2").set_seed(1234).reset()  # noqa
        p1 = l1.parameters()
        p2 = l2.parameters()
        self.assertTrue((p1["linear1"]["weight"] == p2["linear2"]["weight"]).all())  # noqa

    def test_simple_flow(self):
        FEATURES_DIM = 2
        data_len = 100
        batch_size = 32
        epoch_num = 5

        def gen_rand_sample():
            features = np.random.uniform(0, 1, (FEATURES_DIM))
            label = (2 * features).sum() + 0.4
            return Sample.from_ndarray(features, label)

        trainingData = self.sc.parallelize(range(0, data_len)).map(
            lambda i: gen_rand_sample())

        model = Sequential()
        l1 = Linear(FEATURES_DIM, 1, "Xavier").set_name("linear1")
        self.assertEqual("linear1", l1.name())
        model.add(l1)

        state = {"learningRate": 0.01,
                 "learningRateDecay": 0.0002,
                 "learingRateSchedule": Poly(0.5, int((data_len/batch_size)*epoch_num))}  # noqa
        optimizer = Optimizer(
            model=model,
            training_rdd=trainingData,
            criterion=MSECriterion(),
            optim_method="sgd",
            state=state,
            end_trigger=MaxEpoch(epoch_num),
            batch_size=batch_size)
        optimizer.setvalidation(
            batch_size=batch_size,
            val_rdd=trainingData,
            trigger=EveryEpoch(),
            val_method=["Top1Accuracy"]
        )
        tmp_dir = tempfile.mkdtemp()
        optimizer.setcheckpoint(SeveralIteration(1), tmp_dir)
        train_summary = TrainSummary(log_dir=tmp_dir,
                                     app_name="run1")
        train_summary.set_summary_trigger("LearningRate", SeveralIteration(1))
        val_summary = ValidationSummary(log_dir=tmp_dir,
                                        app_name="run1")
        optimizer.set_train_summary(train_summary)
        optimizer.set_val_summary(val_summary)

        trained_model = optimizer.optimize()

        lr_result = train_summary.read_scalar("LearningRate")
        top1_result = val_summary.read_scalar("Top1Accuracy")

        # TODO: add result validation
        parameters = trained_model.parameters()
        self.assertIsNotNone(parameters["linear1"])
        print("parameters %s" % parameters["linear1"])
        predict_result = trained_model.predict(trainingData)
        p = predict_result.take(2)
        print("predict predict: \n")
        for i in p:
            print(str(i) + "\n")
        print(len(p))

        test_results = trained_model.test(trainingData, 32, ["Top1Accuracy"])
        for test_result in test_results:
            print(test_result)


if __name__ == "__main__":
    unittest.main(failfast=True)
