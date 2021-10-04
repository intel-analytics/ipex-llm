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
package com.intel.analytics.zoo.examples.nnframes.imageInference

import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.LoggerFilter
import com.intel.analytics.zoo.pipeline.nnframes._
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.image._
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions._
import scopt.OptionParser

/**
 * Scala example for image classification inference with Caffe Inception model on Spark DataFrame.
 * Please refer to the readme.md in the same folder for more details.
 */
object ImageInferenceExample {
  LoggerFilter.redirectSparkInfoLogs()

  def main(args: Array[String]): Unit = {

    val defaultParams = Utils.LocalParams()
    Utils.parser.parse(args, defaultParams).foreach { params =>
      val sc = NNContext.initNNContext("ImageInference")

      val getImageName = udf { row: Row => row.getString(0)}
      val imageDF = NNImageReader.readImages(params.imagePath, sc,
          resizeH = 256, resizeW = 256, imageCodec = 1)
        .withColumn("imageName", getImageName(col("image")))

      val transformer = RowToImageFeature() -> ImageCenterCrop(224, 224) ->
        ImageChannelNormalize(123, 117, 104) -> ImageMatToTensor() -> ImageFeatureToTensor()

      val model = Module.loadCaffeModel[Float](params.caffeDefPath, params.caffeWeightsPath)
      val dlmodel = NNClassifierModel(model, transformer)
        .setBatchSize(params.batchSize)
        .setFeaturesCol("image")
        .setPredictionCol("prediction")

      val resultDF = dlmodel.transform(imageDF)
      resultDF.select("imageName", "prediction").orderBy("imageName").show(10, false)
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
    batchSize: Int = 16)

  val parser = new OptionParser[LocalParams]("BigDL Example") {
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
  }
}
