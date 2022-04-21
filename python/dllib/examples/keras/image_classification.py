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

from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType, StringType

from bigdl.dllib.nncontext import *
from bigdl.dllib.optim.optimizer import *
from bigdl.dllib.keras.objectives import *
from bigdl.dllib.feature.image import *
from bigdl.dllib.keras.layers import *
from bigdl.dllib.keras.models import *
from bigdl.dllib.nnframes import *


def build_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, 3, 3, input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=[2, 2]))

    model.add(Conv2D(32, 3, 3))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=[2, 2]))

    model.add(Conv2D(64, 3, 3))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=[2, 2]))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    return model


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-b", "--batchSize", type=int, dest="batchSize", default="12")
    parser.add_option("-n", "--maxEpoch", type=int, dest="maxEpoch", default="20")
    parser.add_option("-d", "--dataPath", dest="dataPath")
    (options, args) = parser.parse_args(sys.argv)

    sc = init_nncontext("image classification example")

    imageDF = NNImageReader.readImages(options.dataPath, sc, resizeH=150, resizeW=150)
    getName = udf(lambda row: os.path.basename(row[0]), StringType())
    getLabel = udf(lambda name: 1 if name.startswith('cat') else 2, IntegerType())
    labelDF = imageDF.withColumn("name", getName(col("image"))) \
        .withColumn("label", getLabel(col('name')))
    (trainingDF, validationDF) = labelDF.randomSplit([0.9, 0.1])

    transformers = ImageChannelNormalize(0, 0, 0, 255, 255, 255)
    model = build_model(input_shape=(3, 150, 150))

    model.compile(optimizer="rmsprop",
                  loss="binary_crossentropy",
                  metrics=['accuracy'])

    model.fit(trainingDF, batch_size=options.batchSize, nb_epoch=options.maxEpoch,
              label_cols=["label"], transform=transformers, validation_data=validationDF)
    sc.stop()
