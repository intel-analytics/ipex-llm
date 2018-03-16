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

import re
from bigdl.util.common import *
from bigdl.dlframes.dl_image_reader import *
from bigdl.dlframes.dl_image_transformer import *
from bigdl.transform.vision.image import *
from bigdl.dlframes.dl_classifier import *
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType, StringType
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Need parameters: <model> <imagePath>")
        exit(-1)

    sc = SparkContext(appName="ImageTransferLearningExample", conf=create_spark_conf())
    redire_spark_logs()
    show_bigdl_info_logs()
    init_engine()

    model_path = sys.argv[1]
    image_path = sys.argv[2]
    imageDF = DLImageReader.readImages(image_path, sc)

    # the original dataset contains 25000 images, we only take the first 2400 images for this demo.
    # training data contains 1000 cats images and 1000 dog images (id 1 ~ 1000)
    # validation data contains 200 cats images and 200 dog images (id 1000 ~ 1200)
    getName = udf(lambda row: re.search(r'(cat|dog)\.([\d]*)\.jpg', row[0], re.IGNORECASE).group(0), StringType())
    getID = udf(lambda name: re.search(r'([\d]+)', name).group(0), StringType())
    getLabel = udf(lambda name: 1.0 if name.startswith('cat') else 2.0, DoubleType())
    labelDF = imageDF.withColumn("name", getName(col("image"))) \
        .withColumn("id", getID(col('name'))) \
        .withColumn("label", getLabel(col('name'))) \
        .filter('id<1200')

    transformer = DLImageTransformer(
        Pipeline([Resize(256, 256), CenterCrop(224, 224),
                  ChannelNormalize(123.0, 117.0, 104.0)])
    ).setInputCol("image").setOutputCol("features")
    featureDF = transformer.transform(labelDF)

    preTrainedModel = Model.loadModel(model_path)
    preTrainedDLModel = DLModel(preTrainedModel, [3,224,224])
    embeddingDF = preTrainedDLModel.transform(featureDF).drop('features') \
        .withColumnRenamed('prediction', 'features') \
        .cache()

    trainingDF = embeddingDF.filter('id<1000')
    validationDF = embeddingDF.filter('id>1000 and id<1200')

    lrModel = Sequential().add(Linear(1000, 2)).add(LogSoftMax())
    classifier = DLClassifier(lrModel, ClassNLLCriterion(), [1000]) \
        .setLearningRate(0.003).setBatchSize(40).setMaxEpoch(20)

    catdogModel = classifier.fit(trainingDF)
    predictionDF = catdogModel.transform(validationDF).cache()
    predictionDF.show()

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
    accuracy = evaluator.evaluate(predictionDF)
    # expected error should be less than 10%
    print("Test Error = %g " % (1.0 - accuracy))
