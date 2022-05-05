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
from bigdl.dllib.models.lenet.utils import *
from bigdl.dllib.feature.dataset.transformer import *
from bigdl.dllib.nn.layer import *
from bigdl.dllib.nn.criterion import *
from bigdl.dllib.optim.optimizer import *
from bigdl.dllib.utils.common import *
from bigdl.dllib.nncontext import *
from bigdl.dllib.utils.utils import detect_conda_env_name
import os
from bigdl.dllib.utils.log4Error import *


def build_model(class_num):
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

    # To use MKL-DNN backend, the model has to be a graph model with input and output formats set.
    # Sequential model cannot be used in this case, so we convert it to a graph model.
    if get_bigdl_engine_type() == "MklDnn":
        model = model.to_graph()

        # The format index of input or output format can be checked
        # in: ${BigDL-core}/native-dnn/src/main/java/com/intel/analytics/bigdl/mkl/Memory.java
        model.set_input_formats([7])  # Set input format to nchw
        model.set_output_formats([4])  # Set output format to nc

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
    parser.add_option("--optimizerVersion", dest="optimizerVersion", default="optimizerV1")
    parser.add_option("--cluster-mode", dest="clusterMode", default="local")
    parser.add_option("--mkl-dnn", action="store_true", dest="mklDnn", default=False,
                      help="if enable mkldnn")

    (options, args) = parser.parse_args(sys.argv)

    conf = {}
    if options.mklDnn:
        conf["spark.driver.extraJavaOptions"] = "-Dbigdl.engineType=mkldnn"
        conf["spark.executor.extraJavaOptions"] = "-Dbigdl.engineType=mkldnn"
    if options.clusterMode.startswith("yarn"):
        hadoop_conf = os.environ.get("HADOOP_CONF_DIR")
        invalidInputError(hadoop_conf,
                          "Directory path to hadoop conf not found for yarn-client"
                          "mode.", "Please either specify argument hadoop_conf or"
                          "set the environment variable HADOOP_CONF_DIR")
        spark_conf = create_spark_conf().set("spark.executor.memory", "5g") \
            .set("spark.executor.cores", 2) \
            .set("spark.executor.instances", 2) \
            .set("spark.driver.memory", "2g")
        spark_conf.setAll(conf)

        if options.clusterMode == "yarn-client":
            sc = init_nncontext(spark_conf, cluster_mode="yarn-client", hadoop_conf=hadoop_conf)
        else:
            sc = init_nncontext(spark_conf, cluster_mode="yarn-cluster", hadoop_conf=hadoop_conf)
    elif options.clusterMode == "local":
        spark_conf = SparkConf().set("spark.driver.memory", "10g") \
            .set("spark.driver.cores", 4)
        sc = init_nncontext(spark_conf, cluster_mode="local")
    else:
        invalidInputError(False, "please set cluster_mode as local, yarn-client or yarn-cluster")

    set_optimizer_version(options.optimizerVersion)

    # In order to use MklDnn as the backend, you should:
    # 1. Define a graph model with Model(graph container) or convert a sequential model to a graph
    # model
    # 2. Specify the input and output formats of it.
    #    BigDL needs these format information to build a graph running with MKL-DNN backend
    # 3. Run spark-submit command with correct configurations
    #    --conf "spark.driver.extraJavaOptions=-Dbigdl.engineType=mkldnn"
    #    --conf "spark.executor.extraJavaOptions=-Dbigdl.engineType=mkldnn"

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
        model = Model.loadModel(options.modelPath)
        results = model.evaluate(test_data, options.batchSize, [Top1Accuracy()])
        for result in results:
            print(result)
    sc.stop()
