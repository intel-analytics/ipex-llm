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
package com.intel.analytics.bigdl.example.dlframes.imageTransferLearning

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.dlframes.{DLClassifier, DLModel}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.transform.vision.image._
import com.intel.analytics.bigdl.transform.vision.image.augmentation._
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

import org.apache.spark.ml.{Pipeline, Transformer}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, SQLContext}
import scopt.OptionParser
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import org.apache.spark.SparkContext

object ImageTransferLearning {

  def main(args: Array[String]): Unit = {

    val defaultParams = Utils.LocalParams()

    Utils.parser.parse(args, defaultParams).map { params =>

      val conf = Engine.createSparkConf().setAppName("TransferLearning")
      val sc = SparkContext.getOrCreate(conf)
      val sqlContext = new SQLContext(sc)
      Engine.init

      val createLabel = udf((name: String) => if (name.contains("cat")) 1.0 else 2.0)
      val imagesDF: DataFrame = Utils.loadImages(params.folder, params.batchSize, sqlContext)
        .withColumn("label", createLabel(col("imageName")))
        .withColumnRenamed("features", "imageFeatures")
        .drop("features")

      val Array(validationDF, trainingDF) = imagesDF.randomSplit(Array(0.20, 0.80), seed = 1L)

      validationDF.persist()
      trainingDF.persist()

      val loadedModel = Module
        .loadCaffeModel[Float](params.caffeDefPath, params.modelPath)

      val featurizer = new DLModel[Float](loadedModel, Array(3, 224, 224))
        .setBatchSize(params.batchSize)
        .setFeaturesCol("imageFeatures")
        .setPredictionCol("features")

      val lrModel = Sequential().add(Linear(1000, 2)).add(LogSoftMax())

      val classifier = new DLClassifier(lrModel, ClassNLLCriterion[Float](), Array(1000))
        .setLearningRate(0.003).setBatchSize(params.batchSize)
        .setMaxEpoch(20)

      val pipeline = new Pipeline().setStages(
        Array(featurizer, classifier))

      val pipelineModel = pipeline.fit(trainingDF)
      trainingDF.unpersist()

      val predictions = pipelineModel.transform(validationDF)

      predictions.show(200)
      predictions.printSchema()

      val evaluation = new MulticlassClassificationEvaluator().setPredictionCol("prediction")
        .setMetricName("weightedPrecision").evaluate(predictions)
      println("evaluation result on validationDF: " + evaluation)

      validationDF.unpersist()
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
