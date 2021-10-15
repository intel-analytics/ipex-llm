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

import pytest
from bigdl.dllib.nn.layer import *
from bigdl.dllib.nn.criterion import *
from bigdl.dllib.optim.optimizer import *
from test.bigdl.test_zoo_utils import ZooTestCase
from bigdl.dllib.feature.common import *
from bigdl.dllib.utils.nncontext import init_nncontext, init_spark_conf


class TestFeatureCommon(ZooTestCase):

    def setup_method(self, method):
        """ setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        sparkConf = init_spark_conf().setMaster("local[4]").setAppName("test feature set")
        self.sc = init_nncontext(sparkConf)

    def test_BigDL_adapter(self):
        new_preprocessing = BigDLAdapter(Resize(1, 1))
        assert isinstance(new_preprocessing, Preprocessing)

    def test_relations(self):
        resource_path = os.path.join(os.path.split(__file__)[0], "../resources")
        path = os.path.join(resource_path, "qa")
        relations = Relations.read(path + "/relations.txt")
        assert isinstance(relations, list)
        relations2 = Relations.read(path + "/relations.csv", self.sc, 2)
        assert isinstance(relations2, RDD)
        relations3 = Relations.read_parquet(path + "/relations.parquet", self.sc)
        assert isinstance(relations3, RDD)

    def test_train_FeatureSet(self):

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
        data_set = FeatureSet.image_frame(image_frame).transform(transformer).to_dataset()

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
        assert(predict_result.get_predict().count(), 8)

    def create_feature_set_from_rdd(self):
        dim = 2
        data_len = 100

        def gen_rand_sample():
            features = np.random.uniform(0, 1, dim)
            label = np.array((2 * features).sum() + 0.4)
            return Sample.from_ndarray(features, label)

        FeatureSet.rdd(self.sc.parallelize(range(0, data_len)).map(
            lambda i: gen_rand_sample())).to_dataset()


if __name__ == "__main__":
    pytest.main([__file__])
