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

import os

from bigdl.nn.criterion import *
from bigdl.nn.layer import *
from bigdl.optim.optimizer import Adam
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType, StringType

from zoo.common.nncontext import *
from zoo.feature.image import *
from zoo.pipeline.nnframes import *

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print(sys.argv)
        print("Need parameters: <modelPath> <imagePath>")
        exit(-1)

    sc = init_nncontext("ImageTransferLearningExample ")

    model_path = sys.argv[1]
    image_path = sys.argv[2]
    imageDF = NNImageReader.readImages(image_path, sc, resizeH=300, resizeW=300, image_codec=1)

    getName = udf(lambda row: os.path.basename(row[0]), StringType())
    getLabel = udf(lambda name: 1.0 if name.startswith('cat') else 2.0, DoubleType())
    labelDF = imageDF.withColumn("name", getName(col("image"))) \
        .withColumn("label", getLabel(col('name')))
    (trainingDF, validationDF) = labelDF.randomSplit([0.9, 0.1])

    # compose a pipeline that includes feature transform, pretrained model and Logistic Regression
    transformer = ChainedPreprocessing(
        [RowToImageFeature(), ImageResize(256, 256), ImageCenterCrop(224, 224),
         ImageChannelNormalize(123.0, 117.0, 104.0), ImageMatToTensor(), ImageFeatureToTensor()])

    preTrainedNNModel = NNModel(Model.loadModel(model_path), transformer) \
        .setFeaturesCol("image") \
        .setPredictionCol("embedding")

    lrModel = Sequential().add(Linear(1000, 2)).add(LogSoftMax())
    classifier = NNClassifier(lrModel, ClassNLLCriterion(), SeqToTensor([1000])) \
        .setLearningRate(0.002) \
        .setOptimMethod(Adam()) \
        .setBatchSize(32) \
        .setMaxEpoch(20) \
        .setFeaturesCol("embedding") \
        .setCachingSample(False) \

    pipeline = Pipeline(stages=[preTrainedNNModel, classifier])

    catdogModel = pipeline.fit(trainingDF)
    predictionDF = catdogModel.transform(validationDF).cache()
    predictionDF.sample(False, 0.1).show()

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictionDF)
    # expected error should be less than 10%
    print("Test Error = %g " % (1.0 - accuracy))
