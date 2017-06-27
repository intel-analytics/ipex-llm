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

from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
from bigdl.nn.initialization_method import *
from bigdl.dataset import movielens
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

    def test_training(self):
        cadd = CAdd([5, 1])
        y = np.ones([5, 4])
        bf = np.ones([5, 4])
        for i in range(y.shape[0]):
            bf[i] = i + 1

        def grad_update(mlp, x, y, criterion, learning_rate):
            pred = mlp.forward(x)
            err = criterion.forward(pred, y)
            grad_criterion = criterion.backward(pred, y)
            mlp.zero_grad_parameters()
            mlp.backward(x, grad_criterion)
            mlp.update_parameters(learning_rate)
            return err

        mse = MSECriterion()
        for i in range(0, 1000):
            x = np.random.random((5, 4))
            y = x.copy()
            y = y + bf
            err = grad_update(cadd, x, y, mse, 0.01)
        print(cadd.get_weights()[0])
        self.assertTrue(np.allclose(cadd.get_weights()[0],
                                    np.array([1, 2, 3, 4, 5]).reshape((5, 1)),
                                    rtol=1.e-1))

    def test_load_model(self):
        fc1 = Linear(4, 2)
        fc1.set_weights([np.ones((4, 2)), np.ones((2, ))])
        tmp_path = tempfile.mktemp()
        fc1.save(tmp_path, True)
        fc1_loaded = Model.load(tmp_path)
        self.assertTrue(np.allclose(fc1_loaded.get_weights()[0],
                                    fc1.get_weights()[0]))

    def test_create_node(self):
        import numpy as np
        fc1 = Linear(4, 2)()
        fc2 = Linear(4, 2)()
        cadd = CAddTable()([fc1, fc2])
        output1 = ReLU()(cadd)
        model = Model([fc1, fc2], [output1])
        fc1.element().set_weights([np.ones((4, 2)), np.ones((2, ))])
        fc2.element().set_weights([np.ones((4, 2)), np.ones((2, ))])
        output = model.forward([np.array([0.1, 0.2, -0.3, -0.4]),
                                np.array([0.5, 0.4, -0.2, -0.1])])
        self.assertTrue(np.allclose(output,
                                    np.array([2.2, 2.2])))

    def test_graph_backward(self):
        fc1 = Linear(4, 2)()
        fc2 = Linear(4, 2)()
        cadd = CAddTable()([fc1, fc2])
        output1 = ReLU()(cadd)
        output2 = Threshold(10.0)(cadd)
        model = Model([fc1, fc2], [output1, output2])
        fc1.element().set_weights([np.ones((4, 2)), np.ones((2, ))])
        fc2.element().set_weights([np.ones((4, 2)) * 2, np.ones((2, )) * 2])
        output = model.forward([np.array([0.1, 0.2, -0.3, -0.4]),
                                np.array([0.5, 0.4, -0.2, -0.1])])
        gradInput = model.backward([np.array([0.1, 0.2, -0.3, -0.4]),
                                    np.array([0.5, 0.4, -0.2, -0.1])],
                                   [np.array([1.0, 2.0]),
                                    np.array([3.0, 4.0])])
        self.assertTrue(np.allclose(gradInput[0],
                                    np.array([3.0, 3.0, 3.0, 3.0])))
        self.assertTrue(np.allclose(gradInput[1],
                                    np.array([6.0, 6.0, 6.0, 6.0])))

    def test_load_zip_conf(self):
        from bigdl.util.common import get_bigdl_conf
        import sys
        sys.path = [path for path in sys.path if "spark-bigdl.conf" not in path]
        sys.path.insert(0, os.path.join(os.path.split(__file__)[0], "resources/conf/python-api.zip"))  # noqa
        result = get_bigdl_conf()
        self.assertTrue(result.get("spark.executorEnv.OMP_WAIT_POLICY"), "passive")

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

        model_test = Sequential()
        l1_test = Linear(FEATURES_DIM, 1, "Xavier").set_name("linear1_test")
        self.assertEqual("linear1_test", l1_test.name())
        model_test.add(l1_test)
        model_test.add(Sigmoid())

        model = Sequential()
        l1 = Linear(FEATURES_DIM, 1, "Xavier").set_name("linear1")
        self.assertEqual("linear1", l1.name())
        model.add(l1)

        optim_method = SGD(learningrate=0.01, learningrate_decay=0.0002, weightdecay=0.0,
                           momentum=0.0, dampening=0.0, nesterov=False,
                           leaningrate_schedule=Poly(0.5, int((data_len/batch_size)*epoch_num)))
        optimizer = Optimizer(
            model=model_test,
            training_rdd=trainingData,
            criterion=MSECriterion(),
            optim_method=optim_method,
            end_trigger=MaxEpoch(epoch_num),
            batch_size=batch_size)
        optimizer.set_validation(
            batch_size=batch_size,
            val_rdd=trainingData,
            trigger=EveryEpoch(),
            val_method=["Top1Accuracy"]
        )

        optimizer.optimize()

        optimizer.set_model(model=model)
        tmp_dir = tempfile.mkdtemp()
        optimizer.set_checkpoint(SeveralIteration(1), tmp_dir)
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

    def test_forward_backward(self):
        from bigdl.nn.layer import Linear
        rng = RNG()
        rng.set_seed(100)

        linear = Linear(4, 5)
        input = rng.uniform(0.0, 1.0, [4])
        output = linear.forward(input)
        self.assertTrue(np.allclose(output,
                                    np.array([0.41366524,
                                              0.009532653,
                                              -0.677581,
                                              0.07945433,
                                              -0.5742568]),
                                    atol=1e-6, rtol=0))
        mse = MSECriterion()
        target = rng.uniform(0.0, 1.0, [5])
        loss = mse.forward(output, target)
        print("loss: " + str(loss))
        grad_output = mse.backward(output, rng.uniform(0.0, 1.0, [5]))
        l_grad_output = linear.backward(input, grad_output)

    def test_forward_multiple(self):
        from bigdl.nn.layer import Linear
        rng = RNG()
        rng.set_seed(100)

        input = [rng.uniform(0.0, 0.1, [2]),
                 rng.uniform(0.0, 0.1, [2]) + 0.2]

        grad_output = [rng.uniform(0.0, 0.1, [3]),
                       rng.uniform(0.0, 0.1, [3]) + 0.2]

        linear1 = Linear(2, 3)
        linear2 = Linear(2, 3)

        module = ParallelTable()
        module.add(linear1)
        module.add(linear2)
        module.forward(input)
        module.backward(input, grad_output)

    def test_init_method(self):
        initializers = [
            Zeros(),
            Ones(),
            ConstInitMethod(5),
            RandomUniform(-1, 1),
            RandomNormal(0, 1),
            None
        ]
        special_initializers = [
            Xavier(),
            RandomUniform(),
        ]

        layers = [
            SpatialConvolution(6, 12, 5, 5),
            SpatialShareConvolution(1, 1, 1, 1),
            LookupTable(1, 1, 1e-5, 1e-5, 1e-5, True),
            Bilinear(1, 1, 1, True),
            Cosine(2, 3),
            SpatialFullConvolution(1, 1, 1, 1),
            Add(1),
            Linear(100, 10),
            CMul([1, 2]),
            Mul(),
            PReLU(1),
            Euclidean(1, 1, True),
            SpatialDilatedConvolution(1, 1, 1, 1),
            SpatialBatchNormalization(1),
            BatchNormalization(1, 1e-5, 1e-5, True),
        ]

        special_layers = [
            SpatialConvolution(6, 12, 5, 5),
            SpatialShareConvolution(1, 1, 1, 1),
            Cosine(2, 3),
            SpatialFullConvolution(1, 1, 1, 1),
            Add(1),
            Linear(100, 10),
            CMul([1, 2]),
            Mul(),
            PReLU(1),
            Euclidean(1, 1, True),
            SpatialDilatedConvolution(1, 1, 1, 1),
            SpatialBatchNormalization(1),
            BatchNormalization(1, 1e-5, 1e-5, True),
        ]
        for layer in layers:
            for init1 in initializers:
                for init2 in initializers:
                    layer.set_init_method(init1, init2)

        for layer in special_layers:
            for init1 in special_initializers:
                for init2 in special_initializers:
                    layer.set_init_method(init1, init2)

        SpatialFullConvolution(1, 1, 1, 1).set_init_method(BilinearFiller(), Zeros())

    def test_predict(self):
        np.random.seed(100)
        total_length = 6
        features = np.random.uniform(0, 1, (total_length, 2))
        label = (features).sum() + 0.4
        predict_data = self.sc.parallelize(range(0, total_length)).map(
            lambda i: Sample.from_ndarray(features[i], label))
        model = Linear(2, 1, "Xavier").set_name("linear1").set_seed(1234).reset()
        predict_result = model.predict(predict_data)
        p = predict_result.take(6)
        ground_label = np.array([[-0.47596836], [-0.37598032], [-0.00492062],
                                 [-0.5906958], [-0.12307882], [-0.77907401]], dtype="float32")
        for i in range(0, total_length):
            self.assertTrue(np.allclose(p[i], ground_label[i], atol=1e-6, rtol=0))

    def test_rng(self):
        rng = RNG()
        rng.set_seed(100)
        result = rng.uniform(0.1, 0.2, [2, 3])
        ground_label = np.array([[0.15434049, 0.16711557, 0.12783694],
                                 [0.14120464, 0.14245176, 0.15263824]])
        self.assertTrue(result.shape == (2, 3))
        data = result
        for i in range(0, 2):
            self.assertTrue(np.allclose(data[i], ground_label[i], atol=1e-6, rtol=0))

        rng.set_seed(100)
        result2 = rng.uniform(0.1, 0.2, [2, 3])
        data2 = result2
        for i in range(0, 2):
            self.assertTrue(np.allclose(data[i], data2[i]))

    def test_movielens(self):
        movielens_data = movielens.read_data_sets("/tmp/movielens/")
        ground_label = np.array([[1, 1193], [1, 661]])
        for i in range(0, 2):
            self.assertTrue(np.allclose(movielens_data[i], ground_label[i], atol=1e-6, rtol=0))

if __name__ == "__main__":
    unittest.main(failfast=True)
