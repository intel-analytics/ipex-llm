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

from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
from bigdl.util.common import *
from bigdl.dlframes.dl_image_reader import *
from bigdl.dlframes.dl_image_transformer import *
from bigdl.transform.vision.image import *
from bigdl.dlframes.dl_classifier import *
from bigdl.nn.layer import Model

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Need parameters: <model> <imagePath>", file=sys.stderr)
        exit(-1)

    sc = SparkContext(appName="ImageInferenceExample", conf=create_spark_conf())
    redire_spark_logs()
    show_bigdl_info_logs()
    init_engine()

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    imageDF = DLImageReader.readImages(image_path, sc)
    getName = udf(lambda row: row[0], StringType())
    transformer = DLImageTransformer(
        Pipeline([Resize(256, 256), CenterCrop(224, 224),
                  ChannelNormalize(123.0, 117.0, 104.0),
                  MatToTensor()])
    ).setInputCol("image").setOutputCol("features")
    featureDF = transformer.transform(imageDF).withColumn("name", getName(col("image")))

    model = Model.loadModel(model_path)
    classifierModel = DLClassifierModel(model, [3, 224, 224]).setBatchSize(4)

    predictionDF = classifierModel.transform(featureDF)
    predictionDF.select("name", "prediction").orderBy("name").show(20, False)
