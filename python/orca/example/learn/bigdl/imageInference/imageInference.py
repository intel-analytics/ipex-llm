#
# Copyright 2018 Analytics Zoo Authors.
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

from bigdl.nn.layer import Model
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType, DoubleType

from zoo.feature.image import *
from zoo.pipeline.nnframes import *
from zoo.orca.learn.bigdl.estimator import Estimator
from zoo.orca import init_orca_context, stop_orca_context


def inference(image_path, model_path, batch_size, sc):
    imageDF = NNImageReader.readImages(image_path, sc, resizeH=300, resizeW=300, image_codec=1)
    getName = udf(lambda row: row[0], StringType())
    transformer = ChainedPreprocessing(
        [RowToImageFeature(), ImageResize(256, 256), ImageCenterCrop(224, 224),
         ImageChannelNormalize(123.0, 117.0, 104.0), ImageMatToTensor(), ImageFeatureToTensor()])

    model = Model.loadModel(model_path)

    est = Estimator.from_bigdl(model=model,
                               feature_preprocessing=transformer)

    predictionDF = est.predict(data=imageDF,
                               batch_size=batch_size,
                               feature_cols="image"
                               ).withColumn("name", getName(col("image")))
    return predictionDF


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-m", dest="model_path",
                      help="Required. pretrained model path.")
    parser.add_option("-f", dest="image_path",
                      help="training data path.")
    parser.add_option("--b", "--batch_size", type=int, dest="batch_size", default="56",
                      help="The number of samples per gradient update. Default is 56.")
    parser.add_option('--cluster_mode', type=str, default="local",
                      help='The mode for the Spark cluster. local or yarn.')

    (options, args) = parser.parse_args(sys.argv)

    if not options.model_path:
        parser.print_help()
        parser.error('model_path is required')

    if not options.image_path:
        parser.print_help()
        parser.error('image_path is required')

    cluster_mode = options.cluster_mode
    if cluster_mode == "local":
        sc = init_orca_context(memory="3g")
    elif cluster_mode == "yarn":
        sc = init_orca_context(cluster_mode="yarn-client", num_nodes=2, memory="3g")
    else:
        print("init_orca_context failed. cluster_mode should be either 'local' or 'yarn' but got "
              + cluster_mode)

    image_path = options.image_path
    model_path = options.model_path
    batch_size = options.batch_size

    get_most_possible = udf(lambda p: float(p.index(max(p))), DoubleType())
    predictionDF = inference(image_path, model_path, batch_size, sc) \
        .withColumn("prediction", get_most_possible(col("prediction")))
    predictionDF.select("name", "prediction").orderBy("name").show(20, False)

    print("finished...")
    stop_orca_context()
