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

import com.intel.analytics.bigdl.{tensor, _}
import com.intel.analytics.bigdl.dataset.{DataSet, _}
import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Linear, Sequential}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.{Engine, T}
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

class EstimatorSpec extends FlatSpec with Matchers with BeforeAndAfter {
  val model = new Sequential[Float]()
  var sc : SparkContext = _
  var sQLContext : SQLContext = _
  before {
    Engine.setNodeAndCore(1, 1)
    val conf = Engine.createSparkConf().setAppName("Test Optimizer Wrapper").setMaster("local[4]")
    sc = new SparkContext(conf)
    sQLContext = new SQLContext(sc)
    Engine.init
  }

  "An Estimator" should "works properly" in {

    val inputs = Array[String]("Feature data", "Feature size", "Label data", "Label size")

    var estimator = new DLEstimator[Float]().setInputCols(inputs)

    val featureData = Tensor(2, 10)
    val labelData = Tensor(2, 1)

    val batch = MiniBatch(featureData, labelData)

    val miniBatch = sc.parallelize(Seq(
      MinibatchData[Float](featureData.storage().array(),
        featureData.size(),
        labelData.storage().array(), labelData.size())
    ))


    val model = Linear[Float](10, 1)


    val criterion = ClassNLLCriterion[Float]()

    val paramsTrans = ParamMap(
      estimator.modelTrain -> model,
      estimator.criterion -> criterion,
      estimator.batchShape -> Array(2, 10))

    estimator = estimator.copy(paramsTrans)

    var df : DataFrame = sQLContext.createDataFrame(miniBatch).toDF(inputs : _*)

    val res = estimator.fit(df)

    res.isInstanceOf[MlTransformer] should be(true)

  }

  "An Estimator" should "throws exception without correct inputs" in {

    val inputs = Array[String]("Feature data", "Feature size", "Label data", "Label size")
    var estimator = new DLEstimator[Float]().setInputCols(inputs)

    val featureData = Tensor(2, 10)
    val labelData = Tensor(2, 1)
    val batch = MiniBatch(featureData, labelData)

    val miniBatch = sc.parallelize(Seq(
      MinibatchData[Float](featureData.storage().array(),
        featureData.size(),
        labelData.storage().array(), labelData.size())
    ))

    var df : DataFrame = sQLContext.createDataFrame(miniBatch).toDF(inputs : _*)
    intercept[IllegalArgumentException] {
      val res = estimator.fit(df)
    }

  }

  "An Estimator" should "has same transformate result as Classifier" in {

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val tensorBuffer = new ArrayBuffer[ClassifierDenseVector]()

    val batchSize = 10
    val inputs = Array[String]("Feature data", "Feature size", "Label data", "Label size")

    var estimator = new DLEstimator[Float]().setInputCols(inputs)

    val model = LeNet5(5)

    val batchShape = Array(10, 28, 28)

    var m = 0

    var transformer : DLClassifier[Float] = null

    while (m < 10) {

      val optimizerInput = Tensor[Float](10, 28, 28).apply1(e => Random.nextFloat())

      val optimizerTarget = model.forward(optimizerInput).toTensor[Float]

      val optimizerInputArr = optimizerInput.storage().array()

      val optimizerTargetArr = optimizerTarget.max(2)._2.squeeze().storage().array()

      val criterion = ClassNLLCriterion[Float]()

      val paramsTrans = ParamMap(
        estimator.modelTrain -> model,
        estimator.criterion -> criterion,
        estimator.batchShape -> batchShape)

      estimator = estimator.copy(paramsTrans)

      val miniBatch = sc.parallelize(Seq(
        MinibatchData(optimizerInput.storage().array(),
          optimizerInput.size(),
          optimizerTargetArr, Array(10))
      ))

      val df = sQLContext.createDataFrame(miniBatch).toDF(inputs : _*)

      transformer = estimator.fit(df).asInstanceOf[DLClassifier[Float]]

      m += 1
    }

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
        new DenseVector(classifierInputArr.slice(i * 28 * 28, (i + 1) * 28 * 28).map(_.toDouble))))
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
    transPredicts should be (classifierPredicts)
  }

  after{
    sc.stop()
  }
}

private case class ClassifierDenseVector( val features : DenseVector)

private case class MinibatchData[T](featureData : Array[T], featureSize : Array[Int],
                            labelData : Array[T], labelSize : Array[Int])