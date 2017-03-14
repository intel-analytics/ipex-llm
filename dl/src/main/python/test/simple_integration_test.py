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
from optim.optimizer import *
from util.common import *
import numpy as np
import unittest


class TestWorkFlow(unittest.TestCase):
    def setUp(self):
        sparkConf = create_spark_conf(1, 2)
        self.sc = SparkContext(master="local[*]", appName="test model",
                               conf=sparkConf)
        initEngine(1, 2)

    def tearDown(self):
        self.sc.stop()

    def test_load_bigdl_model(self):
        from dataset.transformer import normalizer
        from dataset import mnist
        import pickle

        data_location = "../test/resources/mnist-data/testing_data.pickle"  # noqa
        with open(data_location, 'r') as dfile:
            (images, labels) = pickle.load(dfile)
            sample_rdd = self.sc.parallelize(images).zip(
                self.sc.parallelize(labels)).map(lambda (features, label):
                                                 Sample.from_ndarray(features,
                                                                     label + 1)).map(
                normalizer(mnist.TEST_MEAN, mnist.TEST_STD))
        model = Model.from_path("../test/resources/pre_trained_lenet/lenet-model.9381")  # noqa
        results = model.test(sample_rdd, 32, ["Top1Accuracy"])
        self.assertEqual(32, results[0].total_num)
        self.assertEqual(1.0, results[0].result)
        self.assertEqual('Top1Accuracy', results[0].method)

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
            optim_method="Adagrad",
            state=state,
            end_trigger=MaxEpoch(epoch_num),
            batch_size=batch_size)
        optimizer.setvalidation(
            batch_size=batch_size,
            val_rdd=trainingData,
            trigger=EveryEpoch(),
            val_method=["Top1Accuracy"]
        )
        optimizer.setcheckpoint(SeveralIteration(1), "/tmp/prototype/")
        train_summary = TrainSummary(log_dir=sc.appName,
        app_name="run1", trigger={"learningRate": "hello"})

        # optimizer.set_train_summary(train_summary)

        trained_model = optimizer.optimize()


        # train_summary.read_scalar("learningRate")


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
    unittest.main()
