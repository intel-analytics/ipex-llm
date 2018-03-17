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
package com.intel.analytics.bigdl.example.dlframes.imageInference

import com.intel.analytics.bigdl.dlframes.{DLClassifierModel, DLModel}
import org.apache.spark.sql.DataFrame
import scopt.OptionParser
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.transform.vision.image.augmentation._
import com.intel.analytics.bigdl.transform.vision.image._
import com.intel.analytics.bigdl.utils.Engine
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext

object ImageInference {

  def main(args: Array[String]): Unit = {

    val defaultParams = Utils.LocalParams()
    Utils.parser.parse(args, defaultParams).map { params =>

      val conf = Engine.createSparkConf().setAppName("ModelInference")
      val sc = SparkContext.getOrCreate(conf)
      val sqlContext = new SQLContext(sc)
      Engine.init

      val imagesDF = Utils.loadImages(params.folder, params.batchSize, sqlContext).cache()

      imagesDF.show(10)
      imagesDF.printSchema()

      val model = Module.loadCaffeModel[Float](params.caffeDefPath, params.modelPath)
      val dlmodel: DLModel[Float] = new DLClassifierModel[Float](
        model, Array(3, 224, 224))
        .setBatchSize(params.batchSize)
        .setFeaturesCol("features")
        .setPredictionCol("prediction")

      val count = imagesDF.count().toInt
      val tranDF = dlmodel.transform(imagesDF.limit(count))

      tranDF.select("imageName", "prediction").show(100, false)
    }
  }
}

object Utils {

  case class LocalParams(caffeDefPath: String = " ",
                         modelPath: String = " ",
                         folder: String = " ",
                         batchSize: Int = 16,
                         nEpochs: Int = 10
                        )

  val defaultParams = LocalParams()

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
      .action((x, c) => c.copy(batchSize = x.toInt))
    opt[Int]('e', "nEpochs")
      .text("epoch numbers")
      .action((x, c) => c.copy(nEpochs = x))
  }

  def loadImages(path: String, partitionNum: Int, sqlContext: SQLContext): DataFrame = {

    val imageFrame: ImageFrame = ImageFrame.read(path, sqlContext.sparkContext)
    val transformer = Resize(256, 256) -> CenterCrop(224, 224) ->
      ChannelNormalize(123, 117, 104, 1, 1, 1) -> MatToTensor() -> ImageFrameToSample()
    val transformed: ImageFrame = transformer(imageFrame)
    val imageRDD = transformed.toDistributed().rdd.map { im =>
      (im.uri, im[Sample[Float]](ImageFeature.sample).getData())
    }
    val imageDF = sqlContext.createDataFrame(imageRDD)
      .withColumnRenamed("_1", "imageName")
      .withColumnRenamed("_2", "features")
    imageDF
  }

}
