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
from bigdl.models.lenet.utils import *
from bigdl.nn.keras.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *


def build_model(class_num):
    model = Sequential()
    model.add(Reshape((1, 28, 28), input_shape=(28, 28, 1)))
    model.add(Convolution2D(32, 3, 3, activation="relu"))
    model.add(Convolution2D(32, 3, 3, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(class_num, activation="softmax"))
    return model


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-b", "--batchSize", type=int, dest="batchSize", default="128")
    parser.add_option("-c", "--checkpointPath", dest="checkpointPath", default="/tmp/lenet5")
    parser.add_option("-t", "--endTriggerType", dest="endTriggerType", default="epoch")
    parser.add_option("-n", "--endTriggerNum", type=int, dest="endTriggerNum", default="20")
    parser.add_option("-d", "--dataPath", dest="dataPath", default="/tmp/mnist")

    (options, args) = parser.parse_args(sys.argv)

    sc = SparkContext(appName="lenet5", conf=create_spark_conf())
    redire_spark_logs()
    show_bigdl_info_logs()
    init_engine()

    (train_data, test_data) = process_mnist_rdd(sc, options)

    optimizer = Optimizer(
        model=build_model(10),
        training_rdd=train_data,
        criterion=ClassNLLCriterion(logProbAsInput=False),
        optim_method=SGD(learningrate=0.01, learningrate_decay=0.0002),
        end_trigger=get_end_trigger(options),
        batch_size=options.batchSize)
    validate_optimizer(optimizer, test_data, options.batchSize, options.checkpointPath)
    trained_model = optimizer.optimize()
    parameters = trained_model.parameters()
    sc.stop()
