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
from bigdl.dataset.transformer import *
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *


def build_model(class_num):
    if get_bigdl_engine_type() == "MklBlas":
        model = Sequential()
        model.add(Reshape([1, 28, 28]))
        model.add(SpatialConvolution(1, 6, 5, 5))
        model.add(Tanh())
        model.add(SpatialMaxPooling(2, 2, 2, 2))
        model.add(SpatialConvolution(6, 12, 5, 5))
        model.add(Tanh())
        model.add(SpatialMaxPooling(2, 2, 2, 2))
        model.add(Reshape([12 * 4 * 4]))
        model.add(Linear(12 * 4 * 4, 100))
        model.add(Tanh())
        model.add(Linear(100, class_num))
        model.add(LogSoftMax())

    else:
        input = Input()
        reshape1 = Reshape([1, 28, 28])(input)
        conv1 = SpatialConvolution(1, 6, 5, 5)(reshape1)
        tanh1 = Tanh()(conv1)
        mp1 = SpatialMaxPooling(2, 2, 2, 2)(tanh1)
        conv2 = SpatialConvolution(6, 12, 5, 5)(mp1)
        tanh2 = Tanh()(conv2)
        mp2 = SpatialMaxPooling(2, 2, 2, 2)(tanh2)
        reshape2 = Reshape([12 * 4 * 4])(mp2)
        linear1 = Linear(12 * 4 * 4, 100)(reshape2)
        tanh3 = Tanh()(linear1)
        linear2 = Linear(100, class_num)(tanh3)
        logsoftmax = LogSoftMax()(linear2)

        model = Model([input], [logsoftmax])

        model.set_input_formats([7]) # Set input format to nchw
        model.set_output_formats([4]) # Set output format to nc

    return model


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-a", "--action", dest="action", default="train")
    parser.add_option("-b", "--batchSize", type=int, dest="batchSize", default="128")
    parser.add_option("-o", "--modelPath", dest="modelPath", default="/tmp/lenet5/model.470")
    parser.add_option("-c", "--checkpointPath", dest="checkpointPath", default="/tmp/lenet5")
    parser.add_option("-t", "--endTriggerType", dest="endTriggerType", default="epoch")
    parser.add_option("-n", "--endTriggerNum", type=int, dest="endTriggerNum", default="20")
    parser.add_option("-d", "--dataPath", dest="dataPath", default="/tmp/mnist")

    (options, args) = parser.parse_args(sys.argv)

    sc = SparkContext(appName="lenet5", conf=create_spark_conf())
    redire_spark_logs()
    show_bigdl_info_logs()
    init_engine()

    if options.action == "train":
        (train_data, test_data) = preprocess_mnist(sc, options)

        optimizer = Optimizer(
            model=build_model(10),
            training_rdd=train_data,
            criterion=ClassNLLCriterion(),
            optim_method=SGD(learningrate=0.01, learningrate_decay=0.0002),
            end_trigger=get_end_trigger(options),
            batch_size=options.batchSize)
        validate_optimizer(optimizer, test_data, options)
        trained_model = optimizer.optimize()
        parameters = trained_model.parameters()
    elif options.action == "test":
        # Load a pre-trained model and then validate it through top1 accuracy.
        test_data = get_mnist(sc, "test", options.dataPath) \
            .map(lambda rec_tuple: (normalizer(rec_tuple[0], mnist.TEST_MEAN, mnist.TEST_STD),
                                    rec_tuple[1])) \
            .map(lambda t: Sample.from_ndarray(t[0], t[1]))
        model = Model.load(options.modelPath)
        results = model.evaluate(test_data, options.batchSize, [Top1Accuracy()])
        for result in results:
            print(result)
    sc.stop()
