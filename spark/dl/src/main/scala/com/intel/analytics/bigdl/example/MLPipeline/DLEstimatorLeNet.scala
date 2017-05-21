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
package com.intel.analytics.bigdl.example.MLPipeline

import java.nio.file.Paths

import com.intel.analytics.bigdl.dataset.{DistributedDataSet, MiniBatch}
import com.intel.analytics.bigdl.dataset.image.{BytesToGreyImg, GreyImgNormalizer, GreyImgToBatch}
import com.intel.analytics.bigdl.models.lenet.{LeNet5, Utils}
import com.intel.analytics.bigdl.models.lenet.Utils._
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.utils.LoggerFilter
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.{DLClassifier, DLEstimator, Pipeline}
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{DataSet, _}
import com.intel.analytics.bigdl.example.imageclassification.MlUtils.{testMean => _, testStd => _, _}
import com.intel.analytics.bigdl.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, T}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import org.apache.spark.SparkContext
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SQLContext}

import scala.collection.mutable.ArrayBuffer

/**
 * An example to show how to use DLEstimator fit to be compatible with ML Pipeline
 * refer to README.md on how to run this example
 */
object DLEstimatorLeNet {

  LoggerFilter.redirectSparkInfoLogs()
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)
  def main(args: Array[String]): Unit = {
    val inputs = Array[String]("Feature data", "Label data")
    trainParser.parse(args, new TrainParams()).map(param => {
      val conf = Engine.createSparkConf()
        .setAppName("MLPipeline Example")
        .set("spark.task.maxFailures", "1")
      val sc = new SparkContext(conf)
      val sqLContext = new SQLContext(sc)
      Engine.init

      val trainData = param.folder + "/train-images-idx3-ubyte"
      val trainLabel = param.folder + "/train-labels-idx1-ubyte"
      val validationData = param.folder + "/t10k-images-idx3-ubyte"
      val validationLabel = param.folder + "/t10k-labels-idx1-ubyte"

      val model = LeNet5(classNum = 10)

      val trainSet = DataSet.array(load(trainData, trainLabel), sc) ->
        BytesToGreyImg(28, 28) -> GreyImgNormalizer(trainMean, trainStd) -> GreyImgToBatch(
        param.batchSize)

      val validationSet = DataSet.array(load(validationData, validationLabel), sc) ->
        BytesToGreyImg(28, 28) -> GreyImgNormalizer(testMean, testStd) -> GreyImgToBatch(
        param.batchSize)

      val dataFrameRDD : RDD[MinibatchData[Float]] = trainSet.
        asInstanceOf[DistributedDataSet[TensorMiniBatch[Float]]].data(false).map(batch => {
          val feature = batch.getInput.asInstanceOf[Tensor[Float]]
          val label = batch.getTarget.asInstanceOf[Tensor[Float]]
          val estimatorData = MinibatchData[Float](feature.storage().array(),
            label.storage().array())
           estimatorData
        })

      var trainingDF : DataFrame = sqLContext.createDataFrame(dataFrameRDD).toDF(inputs : _*)

      val criterion = ClassNLLCriterion[Float]()

      var batchShape = Array[Int](128, 28, 28)

      var estimator = new DLEstimator[Float](model, criterion, batchShape).
        setFeaturesCol(inputs(0)).setLabelCol(inputs(1))

      val paramsTrans = ParamMap(
        estimator.featureSize -> Array(10, 28, 28),
        estimator.labelSize -> Array(10) )

      estimator = estimator.copy(paramsTrans)

      val transformer = estimator.fit(trainingDF).asInstanceOf[DLClassifier[Float]]

      transformer.setInputCol("features")
        .setOutputCol("predict")

      val rdd: RDD[DenseVectorData] = validationSet.
        asInstanceOf[DistributedDataSet[MiniBatch[Float]]].data(false).flatMap{batch => {
          val buffer = new ArrayBuffer[DenseVectorData]()
          val feature = batch.data.storage().toArray
          var i = 0
          while (i < 128) {
            val next = new DenseVector(feature.
             slice(i * 28 * 28, (i + 1) * 28 * 28).map(_.toDouble))
            val data = DenseVectorData(next)
            buffer.append(data)
            i += 1
          }
          buffer.iterator
          }
        }
      var validationDF : DataFrame = sqLContext.createDataFrame(rdd).toDF("features")
      val transformed = transformer.transform(validationDF)
      transformed.select("features", "predict").collect()
        .foreach { case Row(data: DenseVector, predict: Int) =>
            println(data + "=>" + predict)
        }
    })

  }
}

private case class DenseVectorData(denseVector : DenseVector)
private case class MinibatchData[T](featureData : Array[T], labelData : Array[T])
