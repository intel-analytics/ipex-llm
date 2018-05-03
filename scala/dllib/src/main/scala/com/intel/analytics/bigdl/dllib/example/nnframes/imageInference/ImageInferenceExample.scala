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
import com.intel.analytics.zoo.feature.common.{ImageFeatureToTensor, RowToImageFeature}
import com.intel.analytics.zoo.feature.image._
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions._
import scopt.OptionParser

object ImageInferenceExample {
  LoggerFilter.redirectSparkInfoLogs()

  def main(args: Array[String]): Unit = {

    val defaultParams = Utils.LocalParams()
    Utils.parser.parse(args, defaultParams).foreach { params =>
      val sc = NNContext.getNNContext()

      val getImageName = udf { row: Row => row.getString(0)}
      val imageDF = NNImageReader.readImages(params.folder, sc)
        .withColumn("imageName", getImageName(col("image")))

      val transformer = RowToImageFeature() -> Resize(256, 256) -> CenterCrop(224, 224) ->
        ChannelNormalizer(123, 117, 104) -> MatToTensor() -> ImageFeatureToTensor()

      val model = Module.loadCaffeModel[Float](params.caffeDefPath, params.modelPath)
      val dlmodel = new NNClassifierModel(model, transformer)
        .setBatchSize(params.batchSize)
        .setFeaturesCol("image")
        .setPredictionCol("prediction")

      val resultDF = dlmodel.transform(imageDF)
      resultDF.select("imageName", "prediction").show(10, false)
    }
  }
}

private object Utils {

  case class LocalParams(
    caffeDefPath: String = " ",
    modelPath: String = " ",
    folder: String = " ",
    batchSize: Int = 16)

  val parser = new OptionParser[LocalParams]("BigDL Example") {
    opt[String]("caffeDefPath")
      .text(s"caffeDefPath")
      .action((x, c) => c.copy(caffeDefPath = x))
    opt[String]("modelPath")
      .text(s"modelPath")
      .action((x, c) => c.copy(modelPath = x))
    opt[String]("folder")
      .text(s"folder")
      .action((x, c) => c.copy(folder = x))
    opt[Int]('b', "batchSize")
      .text(s"batchSize")
      .action((x, c) => c.copy(batchSize = x))
  }
}
