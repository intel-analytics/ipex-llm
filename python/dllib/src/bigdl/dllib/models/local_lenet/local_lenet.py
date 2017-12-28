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


def get_mnist(data_type="train", location="/tmp/mnist"):
    """
    Get and normalize the mnist data. We would download it automatically
    if the data doesn't present at the specific location.

    :param data_type: training data or testing data
    :param location: Location storing the mnist
    :return: (features: Ndarray, label: Ndarray)
    """
    X, Y = mnist.read_data_sets(location, data_type)
    return X, Y + 1  # The label of ClassNLLCriterion starts from 1 instead of 0


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-b", "--batchSize", type=int, dest="batchSize", default="128")
    parser.add_option("-m", "--max_epoch", type=int, dest="max_epoch", default="20")
    parser.add_option("-d", "--dataPath", dest="dataPath", default="/tmp/mnist")
    (options, args) = parser.parse_args(sys.argv)

    redire_spark_logs()
    show_bigdl_info_logs()
    init_engine()

    (X_train, Y_train) = get_mnist("train", options.dataPath)
    (X_test, Y_test) = get_mnist("test", options.dataPath)

    optimizer = Optimizer.create(
        model=build_model(10),
        training_set=(X_train, Y_train),
        criterion=ClassNLLCriterion(),
        optim_method=SGD(learningrate=0.01, learningrate_decay=0.0002),
        end_trigger=MaxEpoch(options.max_epoch),
        batch_size=options.batchSize)
    optimizer.set_validation(
        batch_size=options.batchSize,
        X_val = X_test,
        Y_val = Y_test,
        trigger=EveryEpoch(),
        val_method=[Top1Accuracy()]
    )
    trained_model = optimizer.optimize()
    predict_result = trained_model.predict_class(X_test)
    print(predict_result)
