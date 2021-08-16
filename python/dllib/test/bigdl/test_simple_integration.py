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

from bigdl.nn.layer import *
from bigdl.nn.initialization_method import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
from bigdl.util.common import _py2java
from bigdl.nn.initialization_method import *
from bigdl.dataset import movielens
import numpy as np
import tempfile
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from bigdl.util.engine import compare_version
from bigdl.transform.vision.image import *
from bigdl.models.utils.model_broadcast import broadcast_model
from bigdl.dataset.dataset import *
np.random.seed(1337)  # for reproducibility


class TestSimple():
    def setup_method(self, method):
        """ setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        sparkConf = create_spark_conf().setMaster("local[4]").setAppName("test model")
        self.sc = get_spark_context(sparkConf)
        init_engine()

    def teardown_method(self, method):
        """ teardown any state that was previously setup with a setup_method
        call.
        """
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

        mse = MSECriterion(self)
        for i in range(0, 1000):
            x = np.random.random((5, 4))
            y = x.copy()
            y = y + bf
            err = grad_update(cadd, x, y, mse, 0.01)
        print(cadd.get_weights()[0])
        assert_allclose(cadd.get_weights()[0],
                        np.array([1, 2, 3, 4, 5]).reshape((5, 1)),

                        rtol=1.e-1)

    def test_load_keras_model_of(self):
        from bigdl.nn.keras.topology import Model as KModel
        from bigdl.nn.keras.layer import Input as KInput
        from bigdl.nn.keras.layer import Dense

        input = KInput(shape=[2, 3])
        fc1 = Dense(2)(input)
        model = KModel(input, fc1)
        tmp_path = tempfile.mktemp()
        model.save(tmp_path, True)
        model_loaded = KModel.load(tmp_path)
        assert "bigdl.nn.keras.topology.Model" in str(type(model_loaded))
        assert len(model_loaded.layers) == 2

    def test_load_keras_seq_of(self):
        from bigdl.nn.keras.topology import Sequential as KSequential
        from bigdl.nn.keras.layer import Dense

        model = KSequential()
        fc1 = Dense(2, input_shape=[2, 3])
        model.add(fc1)
        tmp_path = tempfile.mktemp()
        model.save(tmp_path, True)
        model_loaded = KSequential.load(tmp_path)
        assert "bigdl.nn.keras.topology.Sequential" in str(type(model_loaded))
        assert len(model_loaded.layers) == 1

    def test_load_model_of(self):
        input = Input()
        fc1 = Linear(4, 2)(input)
        model = Model(input, fc1)
        tmp_path = tempfile.mktemp()
        model.save(tmp_path, True)
        model_loaded = Model.load(tmp_path)
        assert "Model" in str(type(model_loaded))
        assert len(model_loaded.layers) == 2

    def test_load_sequential_of(self):
        fc1 = Linear(4, 2)
        model = Sequential()
        model.add(fc1)
        tmp_path = tempfile.mktemp()
        model.save(tmp_path, True)
        model_loaded = Model.load(tmp_path)
        assert "Sequential" in str(type(model_loaded))
        assert len(model_loaded.layers) == 1

    def test_load_model(self):
        fc1 = Linear(4, 2)
        fc1.set_weights([np.ones((4, 2)), np.ones((2,))])
        tmp_path = tempfile.mktemp()
        fc1.save(tmp_path, True)
        fc1_loaded = Model.load(tmp_path)
        assert_allclose(fc1_loaded.get_weights()[0],
                        fc1.get_weights()[0])

    def test_load_model_proto(self):
        fc1 = Linear(4, 2)
        fc1.set_weights([np.ones((4, 2)), np.ones((2,))])
        tmp_path = tempfile.mktemp()
        fc1.saveModel(tmp_path, None, True)
        fc1_loaded = Model.loadModel(tmp_path)
        assert_allclose(fc1_loaded.get_weights()[0],
                        fc1.get_weights()[0])

    def test_load_optim_method(self):
        FEATURES_DIM = 2
        data_len = 100
        batch_size = 32
        epoch_num = 5

        def gen_rand_sample():
            features = np.random.uniform(0, 1, (FEATURES_DIM))
            label = (2 * features).sum() + 0.4
            return Sample.from_ndarray(features, label)

        trainingData = self.sc.parallelize(range(0, data_len)).map(lambda i: gen_rand_sample())
        model = Sequential()
        l1 = Linear(FEATURES_DIM, 1).set_init_method(Xavier(), Zeros()).set_name("linear1")
        model.add(l1)

        sgd = SGD(learningrate=0.01, learningrate_decay=0.0002, weightdecay=0.0,
                  momentum=0.0, dampening=0.0, nesterov=False,
                  leaningrate_schedule=Poly(0.5, int((data_len / batch_size) * epoch_num)))

        tmp_path = tempfile.mktemp()
        sgd.save(tmp_path, True)
        optim_method = OptimMethod.load(tmp_path)
        assert optim_method.learningRate() == sgd.value.learningRate()
        assert optim_method.momentum() == sgd.value.momentum()
        assert optim_method.nesterov() == sgd.value.nesterov()

        optimizer = Optimizer(
            model=model,
            training_rdd=trainingData,
            criterion=MSECriterion(),
            optim_method=optim_method,
            end_trigger=MaxEpoch(epoch_num),
            batch_size=batch_size)
        optimizer.optimize()

    def test_create_node(self):
        import numpy as np
        fc1 = Linear(4, 2)()
        fc2 = Linear(4, 2)()
        cadd = CAddTable()([fc1, fc2])
        output1 = ReLU()(cadd)
        model = Model([fc1, fc2], [output1])
        fc1.element().set_weights([np.ones((4, 2)), np.ones((2,))])
        fc2.element().set_weights([np.ones((4, 2)), np.ones((2,))])
        output = model.forward([np.array([0.1, 0.2, -0.3, -0.4]),
                                np.array([0.5, 0.4, -0.2, -0.1])])
        assert_allclose(output,
                        np.array([2.2, 2.2]))

    def test_graph_backward(self):
        fc1 = Linear(4, 2)()
        fc2 = Linear(4, 2)()
        cadd = CAddTable()([fc1, fc2])
        output1 = ReLU()(cadd)
        output2 = Threshold(10.0)(cadd)
        model = Model([fc1, fc2], [output1, output2])
        fc1.element().set_weights([np.ones((4, 2)), np.ones((2,))])
        fc2.element().set_weights([np.ones((4, 2)) * 2, np.ones((2,)) * 2])
        output = model.forward([np.array([0.1, 0.2, -0.3, -0.4]),
                                np.array([0.5, 0.4, -0.2, -0.1])])
        gradInput = model.backward([np.array([0.1, 0.2, -0.3, -0.4]),
                                    np.array([0.5, 0.4, -0.2, -0.1])],
                                   [np.array([1.0, 2.0]),
                                    np.array([3.0, 4.0])])
        assert_allclose(gradInput[0],
                        np.array([3.0, 3.0, 3.0, 3.0]))
        assert_allclose(gradInput[1],
                        np.array([6.0, 6.0, 6.0, 6.0]))

    def test_get_node(self):
        fc1 = Linear(4, 2)()
        fc2 = Linear(4, 2)()
        fc1.element().set_name("fc1")
        cadd = CAddTable()([fc1, fc2])
        output1 = ReLU()(cadd)
        model = Model([fc1, fc2], [output1])
        res = model.node("fc1")
        assert res.element().name() == "fc1"

    def test_save_graph_topology(self):
        fc1 = Linear(4, 2)()
        fc2 = Linear(4, 2)()
        cadd = CAddTable()([fc1, fc2])
        output1 = ReLU()(cadd)
        output2 = Threshold(10.0)(cadd)
        model = Model([fc1, fc2], [output1, output2])
        model.save_graph_topology(tempfile.mkdtemp())

    def test_graph_preprocessor(self):
        fc1 = Linear(4, 2)()
        fc2 = Linear(4, 2)()
        cadd = CAddTable()([fc1, fc2])
        preprocessor = Model([fc1, fc2], [cadd])
        relu = ReLU()()
        fc3 = Linear(2, 1)(relu)
        trainable = Model([relu], [fc3])
        model = Model(preprocessor, trainable)
        model.forward([np.array([0.1, 0.2, -0.3, -0.4]), np.array([0.5, 0.4, -0.2, -0.1])])
        model.backward([np.array([0.1, 0.2, -0.3, -0.4]), np.array([0.5, 0.4, -0.2, -0.1])],
                       np.array([1.0]))

    def test_load_zip_conf(self):
        from bigdl.util.common import get_bigdl_conf
        import sys
        sys_path_back = sys.path
        sys.path = [path for path in sys.path if "spark-bigdl.conf" not in path]
        sys.path.insert(0, os.path.join(os.path.split(__file__)[0],
                                        "resources/conf/python-api.zip"))  # noqa
        sys.path.insert(0, os.path.join(os.path.split(__file__)[0],
                                        "resources/conf/invalid-python-api.zip"))  # noqa
        result = get_bigdl_conf()
        assert result.get("spark.executorEnv.OMP_WAIT_POLICY"), "passive"
        sys.path = sys_path_back

    def test_set_seed(self):
        w_init = Xavier()
        b_init = Zeros()
        l1 = Linear(10, 20).set_init_method(w_init, b_init).set_name("linear1").set_seed(
            1234).reset()  # noqa
        l2 = Linear(10, 20).set_init_method(w_init, b_init).set_name("linear2").set_seed(
            1234).reset()  # noqa
        p1 = l1.parameters()
        p2 = l2.parameters()
        assert (p1["linear1"]["weight"] == p2["linear2"]["weight"]).all()  # noqa

    def test_simple_flow(self):
        FEATURES_DIM = 2
        data_len = 100
        batch_size = 32
        epoch_num = 5

        def gen_rand_sample():
            features = np.random.uniform(0, 1, (FEATURES_DIM))
            label = np.array((2 * features).sum() + 0.4)
            return Sample.from_ndarray(features, label)

        trainingData = self.sc.parallelize(range(0, data_len)).map(
            lambda i: gen_rand_sample())

        model_test = Sequential()
        l1_test = Linear(FEATURES_DIM, 1).set_init_method(Xavier(), Zeros()) \
            .set_name("linear1_test")
        assert "linear1_test" == l1_test.name()
        model_test.add(l1_test)
        model_test.add(Sigmoid())

        model = Sequential()
        l1 = Linear(FEATURES_DIM, 1).set_init_method(Xavier(), Zeros()).set_name("linear1")
        assert "linear1" == l1.name()
        model.add(l1)

        optim_method = SGD(learningrate=0.01, learningrate_decay=0.0002, weightdecay=0.0,
                           momentum=0.0, dampening=0.0, nesterov=False,
                           leaningrate_schedule=Poly(0.5, int((data_len / batch_size) * epoch_num)))
        optimizer = Optimizer.create(
            model=model_test,
            training_set=trainingData,
            criterion=MSECriterion(),
            optim_method=optim_method,
            end_trigger=MaxEpoch(epoch_num),
            batch_size=batch_size)
        optimizer.set_validation(
            batch_size=batch_size,
            val_rdd=trainingData,
            trigger=EveryEpoch(),
            val_method=[Top1Accuracy()]
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
        optimizer.set_end_when(MaxEpoch(epoch_num * 2))

        trained_model = optimizer.optimize()
        lr_result = train_summary.read_scalar("LearningRate")
        top1_result = val_summary.read_scalar("Top1Accuracy")

        # TODO: add result validation
        parameters = trained_model.parameters()

        assert parameters["linear1"] is not None
        print("parameters %s" % parameters["linear1"])
        predict_result = trained_model.predict(trainingData)
        p = predict_result.take(2)
        print("predict predict: \n")
        for i in p:
            print(str(i) + "\n")
        print(len(p))

        test_results = trained_model.evaluate(trainingData, 32, [Top1Accuracy()])
        for test_result in test_results:
            print(test_result)

    def test_multiple_input(self):
        """
        Test training data of samples with several tensors as feature
        using a sequential model with multiple inputs.
        """
        FEATURES_DIM = 2
        data_len = 100
        batch_size = 32
        epoch_num = 5

        def gen_rand_sample():
            features1 = np.random.uniform(0, 1, (FEATURES_DIM))
            features2 = np.random.uniform(0, 1, (FEATURES_DIM))
            label = np.array((2 * (features1 + features2)).sum() + 0.4)
            return Sample.from_ndarray([features1, features2], label)

        trainingData = self.sc.parallelize(range(0, data_len)).map(
            lambda i: gen_rand_sample())

        model_test = Sequential()
        branches = ParallelTable()
        branch1 = Sequential().add(Linear(FEATURES_DIM, 1)).add(ReLU())
        branch2 = Sequential().add(Linear(FEATURES_DIM, 1)).add(ReLU())
        branches.add(branch1).add(branch2)
        model_test.add(branches).add(CAddTable())

        optim_method = SGD(learningrate=0.01, learningrate_decay=0.0002, weightdecay=0.0,
                           momentum=0.0, dampening=0.0, nesterov=False,
                           leaningrate_schedule=Poly(0.5, int((data_len / batch_size) * epoch_num)))
        optimizer = Optimizer.create(
            model=model_test,
            training_set=trainingData,
            criterion=MSECriterion(),
            optim_method=optim_method,
            end_trigger=MaxEpoch(epoch_num),
            batch_size=batch_size)
        optimizer.set_validation(
            batch_size=batch_size,
            val_rdd=trainingData,
            trigger=EveryEpoch(),
            val_method=[Top1Accuracy()]
        )

        optimizer.optimize()

    def test_table_label(self):
        """
        Test for table as label in Sample.
        For test purpose only.
        """
        def gen_rand_sample():
            features1 = np.random.uniform(0, 1, 3)
            features2 = np.random.uniform(0, 1, 3)
            label = np.array((2 * (features1 + features2)).sum() + 0.4)
            return Sample.from_ndarray([features1, features2], [label, label])

        training_data = self.sc.parallelize(range(0, 50)).map(
            lambda i: gen_rand_sample())

        model_test = Sequential()
        branches = ParallelTable()
        branch1 = Sequential().add(Linear(3, 1)).add(Tanh())
        branch2 = Sequential().add(Linear(3, 1)).add(ReLU())
        branches.add(branch1).add(branch2)
        model_test.add(branches)

        optimizer = Optimizer.create(
            model=model_test,
            training_set=training_data,
            criterion=MarginRankingCriterion(),
            optim_method=SGD(),
            end_trigger=MaxEpoch(5),
            batch_size=32)
        optimizer.optimize()

    def test_forward_backward(self):
        from bigdl.nn.layer import Linear
        rng = RNG()
        rng.set_seed(100)

        linear = Linear(4, 5)
        input = rng.uniform(0.0, 1.0, [4])
        output = linear.forward(input)
        assert_allclose(output,
                        np.array([0.41366524,
                                  0.009532653,
                                  -0.677581,
                                  0.07945433,
                                  -0.5742568]),
                        atol=1e-6, rtol=0)
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
            MsraFiller(False),
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
        model = Linear(2, 1).set_init_method(Xavier(), Zeros()) \
            .set_name("linear1").set_seed(1234).reset()
        predict_result = model.predict(predict_data)
        p = predict_result.take(6)
        ground_label = np.array([[-0.47596836], [-0.37598032], [-0.00492062],
                                 [-0.5906958], [-0.12307882], [-0.77907401]], dtype="float32")
        for i in range(0, total_length):
            assert_allclose(p[i], ground_label[i], atol=1e-6, rtol=0)

        predict_result_with_batch = model.predict(features=predict_data,
                                                  batch_size=4)
        p_with_batch = predict_result_with_batch.take(6)
        for i in range(0, total_length):
            assert_allclose(p_with_batch[i], ground_label[i], atol=1e-6, rtol=0)

        predict_class = model.predict_class(predict_data)
        predict_labels = predict_class.take(6)
        for i in range(0, total_length):
            assert predict_labels[i] == 1

    def test_predict_image(self):
        resource_path = os.path.join(os.path.split(__file__)[0], "resources")
        image_path = os.path.join(resource_path, "pascal/000025.jpg")
        image_frame = ImageFrame.read(image_path, self.sc)
        transformer = Pipeline([Resize(256, 256), CenterCrop(224, 224),
                                ChannelNormalize(0.485, 0.456, 0.406, 0.229, 0.224, 0.225),
                                MatToTensor(), ImageFrameToSample()])
        image_frame.transform(transformer)

        model = Sequential()
        model.add(SpatialConvolution(3, 6, 5, 5))
        model.add(Tanh())

        image_frame = model.predict_image(image_frame)
        predicts = image_frame.get_predict()
        predicts.collect()

    def test_predict_image_local(self):
        resource_path = os.path.join(os.path.split(__file__)[0], "resources")
        image_path = os.path.join(resource_path, "pascal/000025.jpg")
        image_frame = ImageFrame.read(image_path)
        transformer = Pipeline([Resize(256, 256), CenterCrop(224, 224),
                                ChannelNormalize(0.485, 0.456, 0.406, 0.229, 0.224, 0.225),
                                MatToTensor(), ImageFrameToSample()])
        image_frame.transform(transformer)

        model = Sequential()
        model.add(SpatialConvolution(3, 6, 5, 5))
        model.add(Tanh())

        image_frame = model.predict_image(image_frame)
        predicts = image_frame.get_predict()

    def test_rng(self):
        rng = RNG()
        rng.set_seed(100)
        result = rng.uniform(0.1, 0.2, [2, 3])
        ground_label = np.array([[0.15434049, 0.16711557, 0.12783694],
                                 [0.14120464, 0.14245176, 0.15263824]])
        assert result.shape == (2, 3)
        data = result
        for i in range(0, 2):
            assert_allclose(data[i], ground_label[i], atol=1e-6, rtol=0)

        rng.set_seed(100)
        result2 = rng.uniform(0.1, 0.2, [2, 3])
        data2 = result2
        for i in range(0, 2):
            assert_allclose(data[i], data2[i])

    def test_save_jtensor_dict(self):
        tensors = {}
        tensors["tensor1"] = JTensor.from_ndarray(np.random.rand(3, 2))
        tensors["tensor2"] = JTensor.from_ndarray(np.random.rand(3, 2))
        # in old impl, this will throw an exception
        _py2java(self.sc._gateway, tensors)

    def test_compare_version(self):
        assert compare_version("2.1.1", "2.2.0") == -1
        assert compare_version("2.2.0", "1.6.2") == 1
        assert compare_version("2.2.0", "2.2.0") == 0
        assert compare_version("1.6.0", "2.1.0") == -1
        assert compare_version("2.1.0", "2.1.1") == -1
        assert compare_version("2.0.1", "1.5.2") == 1

    def test_local_optimizer_predict(self):
        feature_num = 2
        data_len = 1000
        batch_size = 32
        epoch_num = 500

        X_ = np.random.uniform(0, 1, (data_len, feature_num))
        y_ = (2 * X_).sum(1) + 0.4
        model = Sequential()
        l1 = Linear(feature_num, 1)
        model.add(l1)

        localOptimizer = Optimizer.create(
            model=model,
            training_set=(X_, y_),
            criterion=MSECriterion(),
            optim_method=SGD(learningrate=1e-2),
            end_trigger=MaxEpoch(epoch_num),
            batch_size=batch_size)

        trained_model = localOptimizer.optimize()
        trained_model = model
        w = trained_model.get_weights()
        assert_allclose(w[0], np.array([2, 2]).reshape([1, 2]), rtol=1e-1)
        assert_allclose(w[1], np.array([0.4]), rtol=1e-1)

        predict_result = trained_model.predict_local(X_)
        assert_allclose(y_, predict_result.reshape((data_len,)), rtol=1e-1)

    def test_local_predict_class(self):
        feature_num = 2
        data_len = 3
        X_ = np.random.uniform(-1, 1, (data_len, feature_num))
        model = Sequential()
        l1 = Linear(feature_num, 1)
        model.add(l1)
        model.add(Sigmoid())
        model.set_seed(1234).reset()
        predict_result = model.predict_class(X_)
        assert_array_equal(predict_result, np.ones([3]))

    def test_local_predict_multiple_input(self):
        l1 = Linear(3, 2)()
        l2 = Linear(3, 3)()
        joinTable = JoinTable(dimension=1, n_input_dims=1)([l1, l2])
        model = Model(inputs=[l1, l2], outputs=joinTable)
        result = model.predict_local([np.ones([4, 3]), np.ones([4, 3])])
        assert result.shape == (4, 5)
        result2 = model.predict_class([np.ones([4, 3]), np.ones([4, 3])])
        assert result2.shape == (4,)

        result3 = model.predict_local([JTensor.from_ndarray(np.ones([4, 3])),
                                       JTensor.from_ndarray(np.ones([4, 3]))])
        assert result3.shape == (4, 5)
        result4 = model.predict_class([JTensor.from_ndarray(np.ones([4, 3])),
                                       JTensor.from_ndarray(np.ones([4, 3]))])
        assert result4.shape == (4,)
        result5 = model.predict_local([JTensor.from_ndarray(np.ones([4, 3])),
                                       JTensor.from_ndarray(np.ones([4, 3]))], batch_size=2)
        assert result5.shape == (4, 5)

    def test_model_broadcast(self):

        init_executor_gateway(self.sc)
        model = Linear(3, 2)
        broadcasted = broadcast_model(self.sc, model)
        input_data = np.random.rand(3)
        output = self.sc.parallelize([input_data], 1) \
            .map(lambda x: broadcasted.value.forward(x)).first()
        expected = model.forward(input_data)

        assert_allclose(output, expected)

    def test_train_DataSet(self):
        batch_size = 8
        epoch_num = 5
        images = []
        labels = []
        for i in range(0, 8):
            features = np.random.uniform(0, 1, (200, 200, 3))
            label = np.array([2])
            images.append(features)
            labels.append(label)

        image_frame = DistributedImageFrame(self.sc.parallelize(images),
                                            self.sc.parallelize(labels))

        transformer = Pipeline([BytesToMat(), Resize(256, 256), CenterCrop(224, 224),
                                ChannelNormalize(0.485, 0.456, 0.406, 0.229, 0.224, 0.225),
                                MatToTensor(), ImageFrameToSample(target_keys=['label'])])
        data_set = DataSet.image_frame(image_frame).transform(transformer)

        model = Sequential()
        model.add(SpatialConvolution(3, 1, 5, 5))
        model.add(View([1 * 220 * 220]))
        model.add(Linear(1 * 220 * 220, 20))
        model.add(LogSoftMax())
        optim_method = SGD(learningrate=0.01)
        optimizer = Optimizer.create(
            model=model,
            training_set=data_set,
            criterion=ClassNLLCriterion(),
            optim_method=optim_method,
            end_trigger=MaxEpoch(epoch_num),
            batch_size=batch_size)
        optimizer.set_validation(
            batch_size=batch_size,
            val_rdd=data_set,
            trigger=EveryEpoch(),
            val_method=[Top1Accuracy()]
        )

        trained_model = optimizer.optimize()

        predict_result = trained_model.predict_image(image_frame.transform(transformer))
        assert_allclose(predict_result.get_predict().count(), 8)

    def test_get_node_and_core_num(self):
        node, core = get_node_and_core_number()

        assert node == 1
        assert core == 4

    def tes_read_image_frame(self):
        init_engine()
        resource_path = os.path.join(os.path.split(__file__)[0], "resources")
        image_path = os.path.join(resource_path, "pascal/000025.jpg")
        image_frame = ImageFrame.read(image_path, self.sc)
        count = image_frame.get_image().count()
        assert count == 1

    def test_set_input_output_format(self):
        input1 = Input()
        lstm1 = Recurrent().add(LSTM(128, 128))(input1)
        fc1 = Linear(128, 10)
        t1 = TimeDistributed(fc1)(lstm1)
        model = Model(inputs=input1, outputs=t1)
        model.set_input_formats([4])
        model.set_output_formats([27])


if __name__ == "__main__":
    pytest.main([__file__])
