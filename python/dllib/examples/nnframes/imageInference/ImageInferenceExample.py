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

from bigdl.dllib.nn.layer import Model
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType

from bigdl.dllib.nncontext import *
from bigdl.dllib.feature.image import *
from bigdl.dllib.nnframes import *

from optparse import OptionParser
import sys


def inference(image_path, model_path, batch_size, sc):
    imageDF = NNImageReader.readImages(image_path, sc, resizeH=300, resizeW=300, image_codec=1)
    getName = udf(lambda row: row[0], StringType())
    transformer = ChainedPreprocessing(
        [RowToImageFeature(), ImageResize(256, 256), ImageCenterCrop(224, 224),
         ImageChannelNormalize(123.0, 117.0, 104.0), ImageMatToTensor(), ImageFeatureToTensor()])

    model = Model.loadModel(model_path)
    classifier_model = NNClassifierModel(model, transformer)\
        .setFeaturesCol("image").setBatchSize(batch_size)
    predictionDF = classifier_model.transform(imageDF).withColumn("name", getName(col("image")))
    return predictionDF


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-m", dest="model_path",
                      help="Required. pretrained model path.")
    parser.add_option("-f", dest="image_path",
                      help="training data path.")
    parser.add_option("--b", "--batch_size", type=int, dest="batch_size", default="56",
                      help="The number of samples per gradient update. Default is 56.")
    parser.add_option("--cluster-mode", dest="clusterMode", default="local")

    (options, args) = parser.parse_args(sys.argv)

    if not options.model_path:
        parser.print_help()
        parser.error('model_path is required')

    if not options.image_path:
        parser.print_help()
        parser.error('image_path is required')

    conf = {}
    if options.clusterMode.startswith("yarn"):
        hadoop_conf = os.environ.get("HADOOP_CONF_DIR")
        assert hadoop_conf, "Directory path to hadoop conf not found for yarn-client mode. Please " \
                            "set the environment variable HADOOP_CONF_DIR"
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

    image_path = options.image_path
    model_path = options.model_path
    batch_size = options.batch_size

    predictionDF = inference(image_path, model_path, batch_size, sc)
    predictionDF.select("name", "prediction").orderBy("name").show(20, False)

    print("finished...")
    sc.stop()