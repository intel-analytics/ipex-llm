/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.examples.nnframes.imageTransferLearning

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.Adam
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.zoo.pipeline.nnframes._
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.image._
import org.apache.hadoop.fs.Path
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, Row}
import scopt.OptionParser

/**
 * Scala example for image transfer learning with Caffe Inception model on Spark DataFrame.
 * Please refer to the readme.md in the same folder for more details.
 */
object ImageTransferLearning {

  def main(args: Array[String]): Unit = {

    val defaultParams = Utils.LocalParams()
    Logger.getLogger("org").setLevel(Level.WARN)

    Utils.parser.parse(args, defaultParams).foreach { params =>
      val sc = NNContext.initNNContext()

      val createLabel = udf { row: Row =>
        if (new Path(row.getString(0)).getName.contains("cat")) 1.0 else 2.0
      }
      val imagesDF: DataFrame = NNImageReader.readImages(params.imagePath, sc,
          resizeH = 256, resizeW = 256, imageCodec = 1)
        .withColumn("label", createLabel(col("image")))

      val Array(validationDF, trainingDF) = imagesDF.randomSplit(Array(0.1, 0.9), seed = 42L)

      val transformer = RowToImageFeature() -> ImageCenterCrop(224, 224) ->
        ImageChannelNormalize(123, 117, 104) -> ImageMatToTensor() -> ImageFeatureToTensor()
      val loadedModel = Module.loadCaffeModel[Float](params.caffeDefPath, params.caffeWeightsPath)
      val featurizer = NNModel(loadedModel, transformer)
        .setBatchSize(params.batchSize)
        .setFeaturesCol("image")
        .setPredictionCol("embedding")

      val lrModel = Sequential().add(Linear(1000, 2)).add(LogSoftMax())
      val classifier = NNClassifier(lrModel, ZooClassNLLCriterion[Float](), Array(1000))
        .setFeaturesCol("embedding")
        .setOptimMethod(new Adam[Float]())
        .setLearningRate(0.002)
        .setBatchSize(params.batchSize)
        .setMaxEpoch(params.nEpochs)

      val pipeline = new Pipeline().setStages(Array(featurizer, classifier))
      val pipelineModel = pipeline.fit(trainingDF)
      val predictions = pipelineModel.transform(validationDF).cache()

      predictions.sample(false, 0.1).show(20)
      val evaluation = new MulticlassClassificationEvaluator().setPredictionCol("prediction")
        .setMetricName("weightedPrecision").evaluate(predictions)
      println("evaluation result on validationDF: " + evaluation)
      println("finished...")
      sc.stop()
    }
  }
}


private object Utils {

  case class LocalParams(
    caffeDefPath: String = " ",
    caffeWeightsPath: String = " ",
    imagePath: String = " ",
    batchSize: Int = 32,
    nEpochs: Int = 20)

  val defaultParams = LocalParams()

  val parser = new OptionParser[LocalParams]("Analytics zoo image transfer learning Example") {
    opt[String]("caffeDefPath")
      .text(s"caffeDefPath")
      .action((x, c) => c.copy(caffeDefPath = x))
    opt[String]("caffeWeightsPath")
      .text(s"caffeWeightsPath")
      .action((x, c) => c.copy(caffeWeightsPath = x))
    opt[String]("imagePath")
      .text(s"imagePath")
      .action((x, c) => c.copy(imagePath = x))
    opt[Int]('b', "batchSize")
      .text(s"batchSize")
      .action((x, c) => c.copy(batchSize = x))
    opt[Int]('e', "nEpochs")
      .text("epoch numbers")
      .action((x, c) => c.copy(nEpochs = x))
  }
}
