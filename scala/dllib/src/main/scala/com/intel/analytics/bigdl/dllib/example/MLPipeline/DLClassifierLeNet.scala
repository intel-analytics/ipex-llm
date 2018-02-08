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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.image.{BytesToGreyImg, GreyImgNormalizer, GreyImgToBatch}
import com.intel.analytics.bigdl.dataset.{DataSet, DistributedDataSet, MiniBatch, _}
import com.intel.analytics.bigdl.dlframes.DLClassifier
import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.models.lenet.Utils._
import com.intel.analytics.bigdl.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext

/**
 * An example to show how to use DLEstimator fit to be compatible with ML Pipeline
 * refer to README.md on how to run this example
 */
object DLClassifierLeNet {

  LoggerFilter.redirectSparkInfoLogs()

  def main(args: Array[String]): Unit = {
    val inputs = Array[String]("Feature data", "Label data")
    trainParser.parse(args, new TrainParams()).foreach(param => {
      val conf = Engine.createSparkConf()
        .setAppName("MLPipeline Example")
        .set("spark.task.maxFailures", "1")
      val sc = new SparkContext(conf)
      val sqLContext = SQLContext.getOrCreate(sc)
      Engine.init

      val trainData = param.folder + "/train-images-idx3-ubyte"
      val trainLabel = param.folder + "/train-labels-idx1-ubyte"
      val validationData = param.folder + "/t10k-images-idx3-ubyte"
      val validationLabel = param.folder + "/t10k-labels-idx1-ubyte"

      val trainSet = DataSet.array(load(trainData, trainLabel), sc) ->
        BytesToGreyImg(28, 28) -> GreyImgNormalizer(trainMean, trainStd) -> GreyImgToBatch(1)

      val trainingRDD : RDD[Data[Float]] = trainSet.
        asInstanceOf[DistributedDataSet[MiniBatch[Float]]].data(false).map(batch => {
          val feature = batch.getInput().asInstanceOf[Tensor[Float]]
          val label = batch.getTarget().asInstanceOf[Tensor[Float]]
          Data[Float](feature.storage().array(), label.storage().array())
        })
      val trainingDF = sqLContext.createDataFrame(trainingRDD).toDF(inputs: _*)

      val model = LeNet5(classNum = 10)
      val criterion = ClassNLLCriterion[Float]()
      val featureSize = Array(28, 28)
      val estimator = new DLClassifier[Float](model, criterion, featureSize)
        .setFeaturesCol(inputs(0))
        .setLabelCol(inputs(1))
        .setBatchSize(param.batchSize)
        .setMaxEpoch(param.maxEpoch)
      val transformer = estimator.fit(trainingDF)

      val validationSet = DataSet.array(load(validationData, validationLabel), sc) ->
        BytesToGreyImg(28, 28) -> GreyImgNormalizer(testMean, testStd) -> GreyImgToBatch(1)

      val validationRDD: RDD[Data[Float]] = validationSet.
        asInstanceOf[DistributedDataSet[MiniBatch[Float]]].data(false).map{batch =>
          val feature = batch.getInput().asInstanceOf[Tensor[Float]]
          val label = batch.getTarget().asInstanceOf[Tensor[Float]]
          Data[Float](feature.storage().array(), label.storage().array())
        }
      val validationDF = sqLContext.createDataFrame(validationRDD).toDF(inputs: _*)
      val transformed = transformer.transform(validationDF)
      transformed.show()
      sc.stop()
    })
  }
}

private case class Data[T](featureData : Array[T], labelData : Array[T])
