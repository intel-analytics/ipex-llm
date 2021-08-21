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
from bigdl.dllib.feature.dataset import mnist
from bigdl.dllib.models.lenet.lenet5 import build_model
from bigdl.dllib.nn.criterion import *
from bigdl.dllib.optim.optimizer import *
from bigdl.dllib.utils.common import *


def get_mnist(data_type="train", location="/tmp/mnist"):
    """
    Get mnist dataset with features and label as ndarray.
    Data would be downloaded automatically if it doesn't present at the specific location.

    :param data_type: "train" for training data and "test" for testing data.
    :param location: Location to store mnist dataset.
    :return: (features: ndarray, label: ndarray)
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

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data(options.dataPath)

    # The model used here is exactly the same as the model in ../lenet/lenet5.py
    optimizer = Optimizer.create(
        model=build_model(10),
        training_set=(X_train, Y_train),
        criterion=ClassNLLCriterion(),
        optim_method=SGD(learningrate=0.01, learningrate_decay=0.0002),
        end_trigger=MaxEpoch(options.max_epoch),
        batch_size=options.batchSize)
    optimizer.set_validation(
        batch_size=options.batchSize,
        X_val=X_test,
        Y_val=Y_test,
        trigger=EveryEpoch(),
        val_method=[Top1Accuracy()]
    )
    trained_model = optimizer.optimize()
    predict_result = trained_model.predict_class(X_test)
    print(predict_result)
