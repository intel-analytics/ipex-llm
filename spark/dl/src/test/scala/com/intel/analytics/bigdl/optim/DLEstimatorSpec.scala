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

import com.intel.analytics.bigdl.dataset.MiniBatch
import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Linear, Sequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.sql.{DataFrame, SQLContext}

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

class DLEstimatorSpec extends FlatSpec with Matchers with BeforeAndAfter {
  val model = new Sequential[Float]()
  var sc : SparkContext = null
  var sQLContext : SQLContext = null

  after{
    if (sc != null) {
      sc.stop()
    }
  }

  "An Estimator" should "works properly" in {
    val conf = Engine.createSparkConf().setAppName("Test DLEstimator").setMaster("local[1]")
    sc = SparkContext.getOrCreate(conf)
    sQLContext = new SQLContext(sc)
    Engine.init

    val inputs = Array[String]("Feature data", "Label data")

    val model = Linear[Float](10, 1)

    val criterion = ClassNLLCriterion[Float]()

    var estimator = new DLEstimator[Float](model, criterion, Array(2, 10))
      .setFeaturesCol(inputs(0)).setLabelCol(inputs(1))

    val featureData = Tensor(2, 10)
    val labelData = Tensor(2, 1).fill(1.0f)

    val batch = MiniBatch(featureData, labelData)

    val miniBatch = sc.parallelize(Seq(
      MinibatchData[Float](featureData.storage().array(),
        labelData.storage().array())
    ))

    val paramsTrans = ParamMap(
      estimator.featureSize -> Array(2, 10),
      estimator.labelSize -> Array(2, 1))

    estimator = estimator.copy(paramsTrans)

    var trainingDF: DataFrame = sQLContext.createDataFrame(miniBatch).toDF(inputs: _*)

    val res = estimator.fit(trainingDF)

    res.isInstanceOf[DLTransformer] should be(true)
  }

  "An Estimator" should "throws exception without correct inputs" in {
    val conf = Engine.createSparkConf().setAppName("Test DLEstimator").setMaster("local[1]")
    sc = new SparkContext(conf)
    sQLContext = new SQLContext(sc)
    Engine.init

    val inputs = Array[String]("Feature data", "Label data")

    val model = Linear[Float](10, 1)

    val criterion = ClassNLLCriterion[Float]()

    var estimator = new DLEstimator[Float](model, criterion, Array(2, 10)).
      setFeaturesCol(inputs(0)).setLabelCol(inputs(1))

    val featureData = Tensor(2, 10)
    val labelData = Tensor(2, 1)
    val batch = MiniBatch(featureData, labelData)

    val miniBatch = sc.parallelize(Seq(
      MinibatchData[Float](featureData.storage().array(),
        labelData.storage().array())
    ))

    var df: DataFrame = sQLContext.createDataFrame(miniBatch).toDF(inputs: _*)
    intercept[IllegalArgumentException] {
      val res = estimator.fit(df)
    }
  }

  "An Estimator" should "has same transformate result as Classifier" in {
    val conf = Engine.createSparkConf().setAppName("Test DLEstimator").setMaster("local[1]")
    sc = new SparkContext(conf)
    sQLContext = new SQLContext(sc)
    Engine.init

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val tensorBuffer = new ArrayBuffer[ClassifierDenseVector]()

    val batchSize = 10
    val inputs = Array[String]("Feature data", "Label data")

    val model = LeNet5(5)

    val batchShape = Array(10, 28, 28)

    val criterion = ClassNLLCriterion[Float]()

    var estimator = new DLEstimator[Float](model, criterion, batchShape).
      setFeaturesCol(inputs(0)).setLabelCol(inputs(1))

    var m = 0

    var transformer: DLClassifier[Float] = null

    while (m < 10) {

      val optimizerInput = Tensor[Float](10, 28, 28).apply1(e => Random.nextFloat())

      val optimizerTarget = model.forward(optimizerInput).toTensor[Float]

      val optimizerInputArr = optimizerInput.storage().array()

      val optimizerTargetArr = optimizerTarget.max(2)._2.squeeze().storage().array()

      val paramsTrans = ParamMap(
        estimator.featureSize -> Array(10, 28, 28),
        estimator.labelSize -> Array(10))

      estimator = estimator.copy(paramsTrans)

      val miniBatch = sc.parallelize(Seq(
        MinibatchData(optimizerInput.storage().array(), optimizerTargetArr)
      ))

      val df = sQLContext.createDataFrame(miniBatch).toDF(inputs: _*)

      transformer = estimator.fit(df).asInstanceOf[DLClassifier[Float]]

      m += 1
    }

    transformer.setInputCol("features")
      .setOutputCol("predict")

    val optimizedModel = transformer.getModel

    val transInput = Tensor[Float](10, 28, 28).apply1(e => Random.nextFloat())

    val classifierInput = Tensor[Float]()

    classifierInput.resizeAs(transInput).copy(transInput)

    val transInputDataArr = transInput.storage().array()

    var i = 0
    while (i < batchSize) {
      tensorBuffer.append(new ClassifierDenseVector(
        new DenseVector(transInputDataArr.slice(i * 28 * 28, (i + 1) * 28 * 28).map(_.toDouble))))
      i += 1
    }

    val transRDD = sc.parallelize(tensorBuffer)
    val transDataFrame = sQLContext.createDataFrame(transRDD)

    val transPredicts = transformer.transform(transDataFrame).
      select("predict").collect().map(
      row => {
        row.getAs[Int](0)
      }
    )

    tensorBuffer.clear()

    var classifier = new DLClassifier[Float]()
      .setInputCol("features")
      .setOutputCol("predict")

    val classifierParams = ParamMap(
      classifier.modelTrain -> optimizedModel,
      classifier.batchShape -> batchShape)
    classifier = classifier.copy(classifierParams)

    val classifierInputArr = classifierInput.storage().array()

    i = 0
    while (i < batchSize) {
      tensorBuffer.append(new ClassifierDenseVector(
        new DenseVector(classifierInputArr.slice(i * 28 * 28, (i + 1) * 28 * 28).
          map(_.toDouble))))
      i += 1
    }

    val classifierRDD = sc.parallelize(tensorBuffer)
    val classifierDataFrame = sQLContext.createDataFrame(classifierRDD)

    val classifierPredicts = classifier.transform(classifierDataFrame).
      select("predict").collect().map(
      row => {
        row.getAs[Int](0)
      }
    )
    transPredicts should be(classifierPredicts)
  }

}

private case class ClassifierDenseVector( val features : DenseVector)

private case class MinibatchData[T](featureData : Array[T], labelData : Array[T])
