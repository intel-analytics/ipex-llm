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

package com.intel.analytics.bigdl.dlframes

import java.io.File

import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.{Adam, LBFGS, Loss, Trigger}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.bigdl.visualization.ValidationSummary
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml._
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.mllib.linalg._
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

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

  "An DLClassifierModel" should "return same results after saving and loading" in {
    val data = sqlContext.createDataFrame(smallData).toDF("features", "label")
    val module = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val dlModel = new DLClassifierModel[Float](module, Array(6))
    val result = dlModel.transform(data).rdd.map(_.getAs[Double](2)).collect().sorted

    val filePath = File.createTempFile("DLModel", "bigdl").getPath + Random.nextLong().toString
    dlModel.setIsOverwrite(true)
    dlModel.setBatchSize(10).setFeatureSize(Array(10, 100))
      .setFeaturesCol("test123").setPredictionCol("predict123")
    dlModel.save(filePath)
    val dlModel2 = DLClassifierModel.load(filePath).asInstanceOf[DLClassifierModel[Float]]
    dlModel2.uid shouldEqual dlModel.uid
    dlModel2.getBatchSize shouldEqual dlModel.getBatchSize
    dlModel2.getFeaturesCol shouldEqual dlModel.getFeaturesCol
    dlModel2.getPredictionCol shouldEqual dlModel.getPredictionCol
    dlModel2.getFeatureSize shouldEqual dlModel.getFeatureSize
    dlModel2.setFeatureSize(Array(6)).setFeaturesCol("features").setPredictionCol("prediction")
    val result2 = dlModel2.transform(data).rdd.map(_.getAs[Double](2)).collect().sorted
    result2 shouldEqual result
  }

  "An DLClassifierModel" should "throw Exception when overwriting with isOverWrite = false" in {
    val module = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val dlModel = new DLClassifierModel[Float](module, Array(6))
    val filePath = File.createTempFile("DLModel", "bigdl").getPath + Random.nextLong().toString
    dlModel.setIsOverwrite(false)
    dlModel.save(filePath)
    intercept[Exception] { dlModel.save(filePath) }
    dlModel.setIsOverwrite(true)
    dlModel.save(filePath)
  }

  /**
   * Because package(spark.ml.linalg._) isn't included in spark_1.6,
   * this parts of codes(specialized for 2.0) are commented out by default.
   * Please uncomment them manually if you want to run unit test on spark_2.x.
   */
  "An DLClassifier" should "works in ML pipeline(support both spark_1.6 and spark_2.x)" in {
    val appSparkVersion = org.apache.spark.SPARK_VERSION
    val df: DataFrame = appSparkVersion.trim match {
      case v if v.startsWith("1") =>
        val data = sc.parallelize(smallData.map(p => (new DenseVector(p._1), p._2)))
        sqlContext.createDataFrame(data).toDF("features", "label")
      case v if v.startsWith("2") =>
        val data = sc.parallelize(smallData.map(p => (new linalg.DenseVector(p._1), p._2)))
        sqlContext.createDataFrame(data).toDF("features", "label")
    }
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
    val ppResult = pipelineModel.transform(df).select("label", "prediction")
    assert(ppResult.where("prediction=label").count() > nRecords * 0.8)

    val filePath = File.createTempFile("DLModel", "bigdl").getPath + Random.nextLong().toString
    pipelineModel.save(filePath)
    val ppLoaded = PipelineModel.load(filePath)
    ppLoaded.isInstanceOf[PipelineModel] should be(true)
    val ppLoadedResult = ppLoaded.transform(df).select("label", "prediction")

    val ppPredictionSum = ppResult.collect()
      .map(row => row.getAs[Double](1)).sum
    val ppLoadedPredictionSum = ppLoadedResult.collect()
      .map(row => row.getAs[Double](1)).sum
    ppPredictionSum shouldEqual ppLoadedPredictionSum
  }
}

private case class Data(label: Double, features: Array[Double])
