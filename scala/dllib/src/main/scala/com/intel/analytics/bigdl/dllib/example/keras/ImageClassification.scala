/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.dllib.example.keras

import com.intel.analytics.bigdl.dllib.NNContext
import com.intel.analytics.bigdl.dllib.feature.image.ImageChannelNormalize
import com.intel.analytics.bigdl.dllib.nnframes.NNImageReader
import com.intel.analytics.bigdl.dllib.keras.layers._
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.Sequential
import com.intel.analytics.bigdl.dllib.keras.objectives.BinaryCrossEntropy
import com.intel.analytics.bigdl.dllib.optim._
import com.intel.analytics.bigdl.dllib.models.lenet.Utils._
import com.intel.analytics.bigdl.numeric.NumericFloat
import org.apache.hadoop.fs.Path
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions._

object ImageClassification {
  def buildMode(inputShape: Shape): Sequential[Float] = {
    import com.intel.analytics.bigdl.numeric.NumericFloat
    val model = Sequential()
    model.add(Conv2D(32, 3, 3, inputShape = inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(poolSize = (2, 2)))

    model.add(Conv2D(32, 3, 3))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(poolSize = (2, 2)))

    model.add(Conv2D(64, 3, 3))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(poolSize = (2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    return model
  }

  def main(args: Array[String]): Unit = {
    trainParser.parse(args, new TrainParams()).map(param => {
      val sc = NNContext.initNNContext()

      val createLabel = udf { row: Row =>
        if (new Path(row.getString(0)).getName.contains("cat")) 1 else 2
      }
      val imgDF = NNImageReader.readImages(param.folder, sc, resizeH = 150, resizeW = 150)
        .withColumn("label", createLabel(col("image")))
      val Array(validationDF, trainingDF) = imgDF.randomSplit(Array(0.1, 0.9), seed = 42L)

      val transformers = ImageChannelNormalize(0, 0, 0, 255, 255, 255)
      val model = buildMode(Shape(3, 150, 150))

      val optimMethod = new RMSprop[Float]()

      model.compile(optimizer = optimMethod,
        loss = BinaryCrossEntropy[Float](),
        metrics = List(new Top1Accuracy[Float]()))
      model.fit(trainingDF, batchSize = param.batchSize, nbEpoch = param.maxEpoch,
        labelCols = Array("label"), transform = transformers, valX = validationDF)

      sc.stop()
    })
  }
}
