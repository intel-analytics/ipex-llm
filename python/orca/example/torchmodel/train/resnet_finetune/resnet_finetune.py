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
import os
from optparse import OptionParser
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from pyspark.sql.types import StringType, DoubleType
from pyspark.sql.functions import col, udf
from bigdl.dllib.optim.optimizer import *
from bigdl.dllib.nncontext import *
from bigdl.dllib.feature.image import *
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.torch import TorchModel, TorchLoss
from bigdl.dllib.nnframes import *
from bigdl.dllib.keras.metrics import Accuracy
from bigdl.dllib.utils.utils import detect_conda_env_name


# Define model with Pytorch
class CatDogModel(nn.Module):
    def __init__(self):
        super(CatDogModel, self).__init__()
        self.features = torchvision.models.resnet18(pretrained=True)
        # freeze weight update
        for parameter in self.features.parameters():
            parameter.requires_grad_(False)
        self.dense1 = nn.Linear(1000, 2)

    def forward(self, x):
        # freeze BatchNorm
        self.features.eval()
        x = self.features(x)
        x = F.log_softmax(self.dense1(x), dim=1)
        return x


if __name__ == '__main__':

    if len(sys.argv) == 1:
        print(sys.argv)
        print("Need parameters: <imagePath>")
        exit(-1)
    parser = OptionParser()
    parser.add_option("--executor-cores", type=int, dest="cores", default=4, help="number of executor cores")
    parser.add_option("--num-executors", type=int, dest="executors", default=16, help="number of executors")
    parser.add_option("--executor-memory", type=str, dest="executorMemory", default="30g", help="executor memory")
    parser.add_option("--driver-memory", type=str, dest="driverMemory", default="30g", help="driver memory")
    parser.add_option("--deploy-mode", type=str, dest="deployMode", default="yarn-client", help="yarn deploy mode, yarn-client or yarn-cluster")
    (options, args) = parser.parse_args(sys.argv)

    hadoop_conf = os.environ.get('HADOOP_CONF_DIR')
    assert hadoop_conf, "Directory path to hadoop conf not found for yarn-client mode. Please " \
            "set the environment variable HADOOP_CONF_DIR"

    sc = init_orca_context(cluster_mode=options.deployMode, hadoop_conf=hadoop_conf,
        conf={"spark.executor.memory": options.executorMemory,
                "spark.executor.cores": options.cores,
                "spark.executor.instances": options.executors,
                "spark.driver.memory": options.driverMemory
        })
    model = CatDogModel()
    zoo_model = TorchModel.from_pytorch(model)

    def lossFunc(input, target):
        return nn.NLLLoss().forward(input, target.flatten().long())

    zoo_loss = TorchLoss.from_pytorch(lossFunc)

    # prepare training data as Spark DataFrame
    image_path = sys.argv[1]
    for filepath,dirnames,filenames in os.walk(image_path):
        for filename in filenames:
            print (filename)
    imageDF = NNImageReader.readImages(image_path, sc, resizeH=256, resizeW=256, image_codec=1)
    getName = udf(lambda row: os.path.basename(row[0]), StringType())
    getLabel = udf(lambda name: 1.0 if name.startswith('cat') else 0.0, DoubleType())
    labelDF = imageDF.withColumn("name", getName(col("image"))) \
        .withColumn("label", getLabel(col('name'))).cache()
    (trainingDF, validationDF) = labelDF.randomSplit([0.9, 0.1])

    # run training and evaluation
    featureTransformer = ChainedPreprocessing(
        [RowToImageFeature(), ImageCenterCrop(224, 224),
         ImageChannelNormalize(123.0, 117.0, 104.0, 255.0, 255.0, 255.0),
         ImageMatToTensor(), ImageFeatureToTensor()])

    classifier = NNClassifier(zoo_model, zoo_loss, featureTransformer) \
        .setLearningRate(0.001) \
        .setBatchSize(16) \
        .setMaxEpoch(1) \
        .setFeaturesCol("image") \
        .setCachingSample(False) \
        .setValidation(EveryEpoch(), validationDF, [Accuracy()], 16)

    catdogModel = classifier.fit(trainingDF)

    shift = udf(lambda p: p - 1, DoubleType())
    predictionDF = catdogModel.transform(validationDF) \
        .withColumn("prediction", shift(col('prediction'))).cache()

    correct = predictionDF.filter("label=prediction").count()
    overall = predictionDF.count()
    accuracy = correct * 1.0 / overall

    predictionDF.sample(False, 0.1).show()

    # expecting: accuracy around 95%
    print("Validation accuracy = {}, correct {},  total {}".format(accuracy, correct, overall))
