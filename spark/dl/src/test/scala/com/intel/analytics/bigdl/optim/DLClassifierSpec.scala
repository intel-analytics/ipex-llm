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

import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.SparkContext
import org.apache.spark.ml._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

@deprecated("`DLClassifier` has been migrated to package `com.intel.analytics.bigdl.dlframes`." +
  "This will be removed in BigDL 0.6.", "0.5.0")
class DLClassifierSpec extends FlatSpec with Matchers with BeforeAndAfter {
  var sc : SparkContext = _
  var sqlContext : SQLContext = _
  var smallData: Seq[(Array[Double], Double)] = _
  val nRecords = 100
  val maxEpoch = 20

  before {
    val conf = Engine.createSparkConf().setAppName("Test DLEstimator").setMaster("local[1]")
    sc = SparkContext.getOrCreate(conf)
    sqlContext = new SQLContext(sc)
    Random.setSeed(42)
    RNG.setSeed(42)
    smallData = DLEstimatorSpec.generateTestInput(
      nRecords, Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), -1.0, 42L)
    Engine.init
  }

  after{
    if (sc != null) {
      sc.stop()
    }
  }

  "An DLClassifier" should "has correct default params" in {
    val model = Linear[Float](10, 1)
    val criterion = ClassNLLCriterion[Float]()
    val estimator = new DLClassifier[Float](model, criterion, Array(10))
    assert(estimator.getFeaturesCol == "features")
    assert(estimator.getLabelCol == "label")
    assert(estimator.getMaxEpoch == 50)
    assert(estimator.getBatchSize == 1)
    assert(estimator.getLearningRate == 1e-3)
    assert(estimator.getLearningRateDecay == 0)
  }

  "An DLClassifier" should "get reasonale accuracy" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val classifier = new DLClassifier[Float](model, criterion, Array(6))
      .setOptimMethod(new LBFGS[Float]())
      .setLearningRate(0.1)
      .setBatchSize(nRecords)
      .setMaxEpoch(maxEpoch)
    val data = sc.parallelize(smallData)
    val df = sqlContext.createDataFrame(data).toDF("features", "label")

    val dlModel = classifier.fit(df)
    dlModel.isInstanceOf[DLClassifierModel[_]] should be(true)
    assert(dlModel.transform(df).where("prediction=label").count() > nRecords * 0.8)
  }

  "An DLClassifier" should "support different FEATURE types" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val classifier = new DLClassifier[Float](model, criterion, Array(6))
      .setLearningRate(0.1)
      .setBatchSize(2)
      .setEndWhen(Trigger.maxIteration(2))

    Array(
      sqlContext.createDataFrame(sc.parallelize(smallData.map(p => (p._1, p._2))))
        .toDF("features", "label"), // Array[Double]
      sqlContext.createDataFrame(sc.parallelize(smallData.map(p => (p._1.map(_.toFloat), p._2))))
        .toDF("features", "label"), // Array[Float]
      sqlContext.createDataFrame(sc.parallelize(smallData.map(p => (Vectors.dense(p._1), p._2))))
        .toDF("features", "label") // MLlib Vector
      // TODO: add ML Vector when ut for Spark 2.0+ is ready
    ).foreach { df =>
      val dlModel = classifier.fit(df)
      dlModel.transform(df).collect()
    }
  }

  "An DLClassifier" should "support scalar FEATURE" in {
    val model = new Sequential().add(Linear[Float](1, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val classifier = new DLClassifier[Float](model, criterion, Array(1))
      .setLearningRate(0.1)
      .setBatchSize(2)
      .setEndWhen(Trigger.maxIteration(2))

    Array(
      sqlContext.createDataFrame(sc.parallelize(smallData.map(p => (p._1.head.toFloat, p._2))))
        .toDF("features", "label"), // Float
      sqlContext.createDataFrame(sc.parallelize(smallData.map(p => (p._1.head, p._2))))
        .toDF("features", "label") // Double
      // TODO: add ML Vector when ut for Spark 2.0+ is ready
    ).foreach { df =>
      val dlModel = classifier.fit(df)
      dlModel.transform(df).collect()
    }
  }

  "An DLClassifier" should "fit with adam and LBFGS" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    Seq(new LBFGS[Float], new Adam[Float]).foreach { optimMethod =>
      val classifier = new DLClassifier[Float](model, criterion, Array(6))
        .setBatchSize(nRecords)
        .setMaxEpoch(2)
        .setOptimMethod(optimMethod)
        .setLearningRate(0.1)
      val data = sc.parallelize(smallData)
      val df = sqlContext.createDataFrame(data).toDF("features", "label")
      val dlModel = classifier.fit(df)
      dlModel.isInstanceOf[DLClassifierModel[_]] should be(true)
    }
  }

  "An DLClassifier" should "supports validation data and summary" in {
    val data = sc.parallelize(smallData)
    val df = sqlContext.createDataFrame(data).toDF("features", "label")

    val logdir = com.google.common.io.Files.createTempDir()
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val classifier = new DLClassifier[Float](model, criterion, Array(6))
      .setBatchSize(nRecords)
      .setEndWhen(Trigger.maxIteration(5))
      .setOptimMethod(new Adam[Float])
      .setLearningRate(0.1)
      .setValidation(Trigger.severalIteration(1), df, Array(new Loss[Float]()), 2)
      .setValidationSummary(ValidationSummary(logdir.getPath, "DLEstimatorValidation"))

    classifier.fit(df)
    val validationSummary = classifier.getValidationSummary.get
    val losses = validationSummary.readScalar("Loss")
    validationSummary.close()
    logdir.deleteOnExit()
  }

  "An DLClassifier" should "get the same classification result with BigDL model" in {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val model = LeNet5(10)

    // init
    val valTrans = new DLClassifierModel[Float](model, Array(28, 28))
      .setBatchSize(4)

    val tensorBuffer = new ArrayBuffer[Data]()
    // generate test data with BigDL
    val input = Tensor[Float](10, 28, 28).apply1(e => Random.nextFloat())
    val target = model.forward(input).toTensor[Float]

    // test against DLClassifierModel
    val inputArr = input.storage().array()
    val targetArr = target.max(2)._2.squeeze().storage().array()
    (0 until 10).foreach(i =>
      tensorBuffer.append(
        Data(targetArr(i), inputArr.slice(i * 28 * 28, (i + 1) * 28 * 28).map(_.toDouble))))
    val rowRDD = sc.parallelize(tensorBuffer)
    val testData = sqlContext.createDataFrame(rowRDD)
    assert(valTrans.transform(testData).where("prediction=label").count() == testData.count())
    tensorBuffer.clear()
  }

  "An DLClassifier" should "works in ML pipeline" in {
    var appSparkVersion = org.apache.spark.SPARK_VERSION
    if (appSparkVersion.trim.startsWith("1")) {
      val data = sc.parallelize(
        smallData.map(p => (org.apache.spark.mllib.linalg.Vectors.dense(p._1), p._2)))
      val df: DataFrame = sqlContext.createDataFrame(data).toDF("features", "label")

      val scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaled")
        .setMax(1).setMin(-1)
      val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
      val criterion = ClassNLLCriterion[Float]()
      val estimator = new DLClassifier[Float](model, criterion, Array(6))
        .setBatchSize(nRecords)
        .setOptimMethod(new LBFGS[Float]())
        .setLearningRate(0.1)
        .setMaxEpoch(maxEpoch)
        .setFeaturesCol("scaled")
      val pipeline = new Pipeline().setStages(Array(scaler, estimator))

      val pipelineModel = pipeline.fit(df)
      pipelineModel.isInstanceOf[PipelineModel] should be(true)
      assert(pipelineModel.transform(df).where("prediction=label").count() > nRecords * 0.8)
    }
  }

  "An DLClassifier" should "supports deep copy" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val data = sc.parallelize(
      smallData.map(p => (org.apache.spark.mllib.linalg.Vectors.dense(p._1), p._2)))
    val df: DataFrame = sqlContext.createDataFrame(data).toDF("features", "label")
    val estimator = new DLClassifier[Float](model, criterion, Array(6))
      .setBatchSize(31)
      .setOptimMethod(new LBFGS[Float]())
      .setLearningRate(0.123)
      .setLearningRateDecay(0.432)
      .setMaxEpoch(13)
      .setFeaturesCol("abc")
      .setTrainSummary(new TrainSummary("/tmp", "1"))
      .setValidationSummary(new ValidationSummary("/tmp", "2"))
      .setValidation(Trigger.maxIteration(3), df, Array(new Loss[Float]()), 2)
    val copied = estimator.copy(ParamMap.empty)
    assert(estimator.model ne copied.model)
    assert(estimator.criterion ne copied.criterion)
    assert(estimator.featureSize ne copied.featureSize)

    assert(estimator.model == copied.model)
    assert(estimator.criterion == copied.criterion)
    assert(estimator.featureSize.deep == copied.featureSize.deep)
    assert(estimator.getMaxEpoch == copied.getMaxEpoch)
    assert(estimator.getBatchSize == copied.getBatchSize)
    assert(estimator.getLearningRate == copied.getLearningRate)
    assert(estimator.getLearningRateDecay == copied.getLearningRateDecay)
    assert(estimator.getFeaturesCol == copied.getFeaturesCol)
    assert(estimator.getLabelCol == copied.getLabelCol)
    val estVal = estimator.getValidation.get
    val copiedVal = copied.getValidation.get
    assert(estVal._1 == copiedVal._1)
    assert(estVal._2 == copiedVal._2)
    assert(estVal._3.deep == copiedVal._3.deep)
    assert(estVal._4 == copiedVal._4)

    // train Summary and validation Summary are not copied since they are not thread-safe and cannot
    // be shared among estimators
    assert(copied.getTrainSummary.isEmpty)
    assert(copied.getTrainSummary.isEmpty)
  }

  "A DLClassifierModel" should "supports deep copy" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val data = sc.parallelize(
      smallData.map(p => (org.apache.spark.mllib.linalg.Vectors.dense(p._1), p._2)))
    val df: DataFrame = sqlContext.createDataFrame(data).toDF("abc", "la")
    val estimator = new DLClassifier[Float](model, criterion, Array(6))
      .setBatchSize(31)
      .setOptimMethod(new LBFGS[Float]())
      .setLearningRate(0.123)
      .setLearningRateDecay(0.432)
      .setMaxEpoch(3)
      .setFeaturesCol("abc")
      .setLabelCol("la")

    val dlModel = estimator.fit(df)
    val copied = dlModel.copy(ParamMap.empty)
    assert(copied.isInstanceOf[DLClassifierModel[Float]])
    assert(dlModel.model ne copied.model)
    assert(dlModel.featureSize ne copied.featureSize)

    assert(dlModel.model == copied.model)
    assert(dlModel.featureSize.deep == copied.featureSize.deep)
    assert(dlModel.getBatchSize == copied.getBatchSize)
    assert(dlModel.getLearningRate == copied.getLearningRate)
    assert(dlModel.getMaxEpoch == copied.getMaxEpoch)
    assert(dlModel.getLearningRateDecay == copied.getLearningRateDecay)
    assert(dlModel.getFeaturesCol == copied.getFeaturesCol)
  }
}

private case class Data(label: Double, features: Array[Double])
