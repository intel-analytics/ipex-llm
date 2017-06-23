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

package com.intel.analytics.bigdl.optim

import scala.util.Random

import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Linear, Sequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkContext, SparkException}
import org.apache.spark.ml.{DLClassifier, DLEstimator, DLModel}
import org.apache.spark.sql.{DataFrame, SQLContext}

class DLEstimatorSpec extends FlatSpec with Matchers with BeforeAndAfter {
  val model = new Sequential[Float]()
  var sc : SparkContext = null
  var sqlContext : SQLContext = null

  before {
    val conf = Engine.createSparkConf().setAppName("Test DLEstimator").setMaster("local[1]")
    sc = SparkContext.getOrCreate(conf)
    sqlContext = new SQLContext(sc)
    Engine.init
  }

  after{
    if (sc != null) {
      sc.stop()
    }
  }

  "An Estimator" should "works properly" in {

    val model = Linear[Float](10, 1)
    val criterion = ClassNLLCriterion[Float]()
    val inputs = Array[String]("Feature data", "Label data")
    var estimator = new DLEstimator[Float](model, criterion, Array(10), Array(1))
      .setBatchSize(2)
      .setFeaturesCol(inputs(0))
      .setLabelCol(inputs(1))
      .setMaxEpoch(2)

    val featureData = Tensor(10)
    val labelData = Tensor(1).fill(1.0f)
    val miniBatch = sc.parallelize(Seq(
      MinibatchData[Float](featureData.storage().array(), labelData.storage().array())
    ))
    var trainingDF: DataFrame = sqlContext.createDataFrame(miniBatch).toDF(inputs: _*)

    val res = estimator.fit(trainingDF)
    res.isInstanceOf[DLModel[_]] should be(true)
  }

  "An Estimator" should "throws exception without correct inputs" in {

    val model = Linear[Float](10, 1)
    val criterion = ClassNLLCriterion[Float]()
    val inputs = Array[String]("Feature data", "Label data")
    var estimator = new DLEstimator[Float](model, criterion, Array(10), Array(2, 1)).
      setFeaturesCol(inputs(0)).setLabelCol(inputs(1))

    val featureData = Tensor(2, 10)
    val labelData = Tensor(2, 1)
    val miniBatch = sc.parallelize(Seq(
      MinibatchData[Float](featureData.storage().array(), labelData.storage().array())
    ))
    var df: DataFrame = sqlContext.createDataFrame(miniBatch).toDF(inputs: _*)

    // Spark 1.6 and 2.0 throws different exception here
    intercept[Exception] {
      estimator.fit(df)
    }
  }

  "An Estimator" should "has same transform result as Classifier" in {

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val batchSize = 2
    val inputs = Array[String]("Feature data", "Label data")
    val featureSize = Array(28, 28)

    val model = LeNet5(5)
    val criterion = ClassNLLCriterion[Float]()
    var estimator = new DLEstimator[Float](model, criterion, featureSize, Array(1))
      .setFeaturesCol(inputs(0))
      .setLabelCol(inputs(1))
      .setBatchSize(batchSize)

    val optimizerInput = Tensor[Float](28, 28).apply1(e => Random.nextFloat())
    val optimizerTarget = model.forward(optimizerInput).toTensor[Float]
    val optimizerTargetArr = optimizerTarget.max(1)._2.squeeze().storage().array()
    val miniBatch = sc.parallelize( Seq(
      MinibatchData(optimizerInput.storage().array(), optimizerTargetArr),
      MinibatchData(optimizerInput.storage().array(), optimizerTargetArr)
    ))
    val df = sqlContext.createDataFrame(miniBatch).toDF(inputs: _*)

    val dlModel = estimator.fit(df).asInstanceOf[DLModel[Float]]
    val transPredicts = dlModel.transform(df).select("prediction").collect().map { row =>
        row.getSeq[Double](0).head
      }

    val classifier = new DLClassifier[Float](dlModel.model, featureSize)
      .setFeaturesCol(inputs(0))
      .setBatchSize(batchSize)
    val classifierPredicts = classifier.transform(df).
      select("prediction").collect().map(
      row => {
        row.getSeq[Double](0).head
      }
    )
    transPredicts should be(classifierPredicts)
  }
}

private case class MinibatchData[T](featureData : Array[T], labelData : Array[T])
