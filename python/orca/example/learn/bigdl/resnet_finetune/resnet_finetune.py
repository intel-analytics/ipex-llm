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

import argparse
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from pyspark.sql.types import StringType, DoubleType
from pyspark.sql.functions import col, udf
from bigdl.optim.optimizer import *

from zoo.feature.image import *
from zoo.orca.learn.metrics import Accuracy
from zoo.pipeline.api.torch import TorchModel, TorchLoss
from zoo.pipeline.nnframes import *

from zoo.orca.learn.bigdl.estimator import Estimator
from zoo.orca import init_orca_context, stop_orca_context


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

    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster_mode', type=str, default="local",
                        help='The mode for the Spark cluster. local or yarn.')
    parser.add_argument('--imagePath', type=str,
                        help='The path to your train samples.')
    args = parser.parse_args()

    cluster_mode = args.cluster_mode

    if cluster_mode == "local":
        num_cores_per_executor = 4
        sc = init_orca_context(cores=num_cores_per_executor, conf={"spark.driver.memory": "2g"})
    elif cluster_mode == "yarn":
        num_executors = 2
        num_cores_per_executor = 4
        hadoop_conf_dir = os.environ.get('HADOOP_CONF_DIR')
        sc = init_orca_context(cluster_mode="yarn-client", cores=num_cores_per_executor,
                               memory="8g", num_nodes=num_executors, driver_memory="2g",
                               driver_cores=1, hadoop_conf=hadoop_conf_dir)
    else:
        print("init_orca_context failed. cluster_mode should be either 'local' or 'yarn' but got "
              + cluster_mode)

    model = CatDogModel()
    zoo_model = TorchModel.from_pytorch(model)

    def lossFunc(input, target):
        return nn.NLLLoss().forward(input, target.flatten().long())

    zoo_loss = TorchLoss.from_pytorch(lossFunc)

    # prepare training data as Spark DataFrame
    image_path = args.imagePath
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

    est = Estimator.from_bigdl(model=zoo_model,
                               loss=zoo_loss,
                               optimizer=SGD(learningrate=0.001),
                               feature_preprocessing=featureTransformer,
                               metrics=Accuracy())
    est.fit(data=trainingDF,
            batch_size=16,
            epochs=1,
            feature_cols="image",
            caching_sample=False,
            validation_data=validationDF,
            validation_trigger=EveryEpoch()
            )

    shift = udf(lambda p: float(p.index(max(p))), DoubleType())
    predictionDF = est.predict(data=validationDF, feature_cols="image") \
        .withColumn("prediction", shift(col('prediction'))).cache()

    correct = predictionDF.filter("label=prediction").count()
    overall = predictionDF.count()
    accuracy = correct * 1.0 / overall

    predictionDF.sample(False, 0.1).show()

    # expecting: accuracy around 95%
    print("Validation accuracy = {}, correct {},  total {}".format(accuracy, correct, overall))
    stop_orca_context()
