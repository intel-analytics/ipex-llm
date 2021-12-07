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

from bigdl.dllib.nncontext import *
from bigdl.dllib.keras.autograd import *
from bigdl.dllib.keras.layers import *
from bigdl.dllib.keras.models import *
from optparse import OptionParser


def mean_absolute_error(y_true, y_pred):
    result = mean(abs(y_true - y_pred), axis=1)
    return result


def add_one_func(x):
    return x + 1.0


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--nb_epoch", dest="nb_epoch", default="5")
    parser.add_option("--batch_size", type=int,dest="batch_size", default=512)
    parser.add_option("--cluster-mode", dest="clusterMode", default="local")

    (options, args) = parser.parse_args(sys.argv)

    conf = {}
    if options.clusterMode.startswith("yarn"):
        hadoop_conf = os.environ.get("HADOOP_CONF_DIR")
        assert hadoop_conf, "Directory path to hadoop conf not found for yarn-client mode. Please " \
                "set the environment variable HADOOP_CONF_DIR"
        spark_conf = create_spark_conf().set("spark.executor.memory", "5g") \
            .set("spark.executor.cores", 2) \
            .set("spark.executor.instances", 2) \
            .set("spark.driver.memory", "4g")
        spark_conf.setAll(conf)

        if options.clusterMode == "yarn-client":
            sc = init_nncontext(spark_conf, cluster_mode="yarn-client", hadoop_conf=hadoop_conf)
        else:
            sc = init_nncontext(spark_conf, cluster_mode="yarn-cluster", hadoop_conf=hadoop_conf)
    elif options.clusterMode == "local":
        spark_conf = SparkConf().set("spark.driver.memory", "10g")\
            .set("spark.driver.cores", 4)
        sc = init_nncontext(spark_conf, cluster_mode="local")
    elif options.clusterMode == "spark-submit":
        sc = init_nncontext(cluster_mode="spark-submit")

    data_len = 1000
    X_ = np.random.uniform(0, 1, (1000, 2))
    Y_ = ((2 * X_).sum(1) + 0.4).reshape([data_len, 1])

    a = Input(shape=(2,))
    b = Dense(1)(a)
    c = Lambda(function=add_one_func)(b)
    model = Model(input=a, output=c)

    model.compile(optimizer=SGD(learningrate=1e-2),
                  loss=mean_absolute_error)

    model.set_tensorboard('./log', 'customized layer and loss')

    model.fit(x=X_,
              y=Y_,
              batch_size=options.batch_size,
              nb_epoch=int(options.nb_epoch),
              distributed=True)

    model.save_graph_topology('./log')

    w = model.get_weights()
    print(w)
    pred = model.predict_local(X_)
    print("finished...")
    sc.stop()
