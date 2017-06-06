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

from optparse import OptionParser

from bigdl.dataset import mnist
from bigdl.dataset.transformer import *
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *


def build_model(class_num):
    model = Sequential()
    model.add(Reshape([1, 28, 28]))
    model.add(SpatialConvolution(1, 6, 5, 5))
    model.add(Tanh())
    model.add(SpatialMaxPooling(2, 2, 2, 2))
    model.add(Tanh())
    model.add(SpatialConvolution(6, 12, 5, 5))
    model.add(SpatialMaxPooling(2, 2, 2, 2))
    model.add(Reshape([12 * 4 * 4]))
    model.add(Linear(12 * 4 * 4, 100))
    model.add(Tanh())
    model.add(Linear(100, class_num))
    model.add(LogSoftMax())
    return model


def get_minst(sc, data_type="train", location="/tmp/mnist"):
    """
    Get and normalize the mnist data. We would download it automatically
    if the data doesn't present at the specific location.
    :param sc: SparkContext
    :param data_type: training data or testing data
    :param location: Location storing the mnist
    :return: A RDD of Sample
    """
    (images, labels) = mnist.read_data_sets(location, data_type)
    images = sc.parallelize(images)
    labels = sc.parallelize(labels)
    # Target start from 1 in BigDL
    record = images.zip(labels).map(lambda features_label:
                                    Sample.from_ndarray(features_label[0], features_label[1] + 1))
    return record


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-a", "--action", dest="action", default="train")
    parser.add_option("-b", "--batchSize", type=int, dest="batchSize", default="128")
    parser.add_option("-m", "--modelPath", dest="modelPath", default="")

    (options, args) = parser.parse_args(sys.argv)

    sc = SparkContext(appName="lenet5", conf=create_spark_conf())
    init_engine()

    if options.action == "train":
        train_data = get_minst(sc, "train").map(
            normalizer(mnist.TRAIN_MEAN, mnist.TRAIN_STD))
        test_data = get_minst(sc, "test").map(
            normalizer(mnist.TEST_MEAN, mnist.TEST_STD))
        optimizer = Optimizer(
            model=build_model(10),
            training_rdd=train_data,
            criterion=ClassNLLCriterion(),
            optim_method=SGD(learningrate=0.01, learningrate_decay=0.0002),
            end_trigger=MaxEpoch(20),
            batch_size=options.batchSize)
        optimizer.set_validation(
            batch_size=options.batchSize,
            val_rdd=test_data,
            trigger=EveryEpoch(),
            val_method=["Top1Accuracy"]
        )
        optimizer.set_checkpoint(EveryEpoch(), "/tmp/lenet5/")
        trained_model = optimizer.optimize()
        parameters = trained_model.parameters()
    elif options.action == "test":
        # Load a pre-trained model and then validate it through top1 accuracy.
        test_data = get_minst(sc, "test").map(
            normalizer(mnist.TEST_MEAN, mnist.TEST_STD))
        model = Model.load(options.modelPath)
        results = model.test(test_data, options.batchSize, ["Top1Accuracy"])
        for result in results:
            print(result)
