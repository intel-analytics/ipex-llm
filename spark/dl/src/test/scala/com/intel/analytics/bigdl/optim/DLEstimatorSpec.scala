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

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.{DLEstimator, DLModel, Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.util.Random

class DLEstimatorSpec extends FlatSpec with Matchers with BeforeAndAfter {
  val model = new Sequential[Float]()
  var sc : SparkContext = _
  var sqlContext : SQLContext = _
  var smallData: Seq[(Array[Double], Double)] = _
  val nRecords = 100
  val maxEpoch = 20

  before {
    Random.setSeed(42)
    RNG.setSeed(42)
    val conf = Engine.createSparkConf().setAppName("Test DLEstimator").setMaster("local[1]")
    sc = SparkContext.getOrCreate(conf)
    sqlContext = new SQLContext(sc)
    smallData = DLEstimatorSpec.generateTestInput(
      nRecords, Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), -1.0, 42L)
    Engine.init
  }

  after{
    if (sc != null) {
      sc.stop()
    }
  }

  "An DLEstimator" should "has correct default params" in {
    val model = Linear[Float](10, 1)
    val criterion = ClassNLLCriterion[Float]()
    val estimator = new DLEstimator[Float](model, criterion, Array(10), Array(1))
    assert(estimator.getFeaturesCol == "features")
    assert(estimator.getLabelCol == "label")
    assert(estimator.getMaxEpoch == 100)
    assert(estimator.getBatchSize == 1)
  }

  "An DLEstimator" should "fit on feature(one dimension Array[Double]) and label(Double)" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val estimator = new DLEstimator[Float](model, criterion, Array(6), Array(1))
      .setBatchSize(nRecords)
      .setMaxEpoch(maxEpoch)
    val data = sc.parallelize(smallData)
    val df = sqlContext.createDataFrame(data).toDF("features", "label")

    val dlModel = estimator.fit(df)
    dlModel.isInstanceOf[DLModel[_]] should be(true)
    val correct = dlModel.transform(df).select("label", "prediction").rdd.filter {
      case Row(label: Double, prediction: Seq[Double]) =>
        label == prediction.indexOf(prediction.max) + 1
    }.count()
    assert(correct > nRecords * 0.8)
  }

  "An DLEstimator" should "fit on feature(one dimension Array[Float]) and label(Double)" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val estimator = new DLEstimator[Float](model, criterion, Array(6), Array(1))
      .setBatchSize(nRecords)
      // intentionally set low since this only validates data format compatibitliy
      .setMaxEpoch(1)
    val data = sc.parallelize(smallData.map(p => (p._1.map(_.toFloat), p._2)))
    val df: DataFrame = sqlContext.createDataFrame(data).toDF("features", "label")

    val dlModel = estimator.fit(df)
    dlModel.isInstanceOf[DLModel[_]] should be(true)
    dlModel.transform(df).count()
  }

  "An DLEstimator" should
    "fit on feature(one dimension Array[Double]) and label(Array[Double])" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = MultiLabelSoftMarginCriterion[Float]()
    val estimator = new DLEstimator[Float](model, criterion, Array(6), Array(2))
      // intentionally set low since this only validates data format compatibitliy
      .setMaxEpoch(1)
      .setBatchSize(nRecords)
    val data = sc.parallelize(
      smallData.map(p => (p._1, if (p._2 == 1.0) Array(0.0, 1.0) else Array(1.0, 0.0))))
    val df: DataFrame = sqlContext.createDataFrame(data).toDF("features", "label")

    val dlModel = estimator.fit(df)
    dlModel.isInstanceOf[DLModel[_]] should be(true)
    dlModel.transform(df).count()
  }

  "An DLEstimator" should "fit on feature(one dimension Array[Float]) and label(Array[Float])" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = MultiLabelSoftMarginCriterion[Float]()
    val estimator = new DLEstimator[Float](model, criterion, Array(6), Array(2))
      // intentionally set low since this only validates data format compatibitliy
      .setMaxEpoch(1)
      .setBatchSize(nRecords)
    val data = sc.parallelize(
      smallData.map(p => (p._1.map(_.toFloat),
        if (p._2 == 1.0) Array(0.0f, 1.0f) else Array(1.0f, 0.0f))))
    val df: DataFrame = sqlContext.createDataFrame(data).toDF("features", "label")

    val dlModel = estimator.fit(df)
    dlModel.isInstanceOf[DLModel[_]] should be(true)
    dlModel.transform(df).count()
  }

  "An DLEstimator" should "work with tensor data" in {

    val model = Linear[Float](10, 1)
    val criterion = ClassNLLCriterion[Float]()
    val estimator = new DLEstimator[Float](model, criterion, Array(10), Array(1))
      .setMaxEpoch(1)
      .setBatchSize(nRecords)

    val featureData = Tensor(10)
    val labelData = Tensor(1).fill(1.0f)
    val miniBatch = sc.parallelize(Seq(
      MinibatchData[Float](featureData.storage().array(), labelData.storage().array())
    ))
    val trainingDF: DataFrame = sqlContext.createDataFrame(miniBatch).toDF("features", "label")

    val dlModel = estimator.fit(trainingDF)
    dlModel.isInstanceOf[DLModel[_]] should be(true)
    dlModel.transform(trainingDF).collect()
  }

  "An DLEstimator" should "work with MLlib Vectors" in {
    var appSparkVersion = org.apache.spark.SPARK_VERSION
    if (appSparkVersion.trim.startsWith("1")) {
      val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
      val criterion = ClassNLLCriterion[Float]()
      val estimator = new DLEstimator[Float](model, criterion, Array(6), Array(1))
        // intentionally set low since this only validates data format compatibitliy
        .setMaxEpoch(1)
        .setBatchSize(nRecords)
      val data = sc.parallelize(
        smallData.map(p => (org.apache.spark.mllib.linalg.Vectors.dense(p._1), p._2)))
      val df: DataFrame = sqlContext.createDataFrame(data).toDF("features", "label")

      val dlModel = estimator.fit(df)
      dlModel.isInstanceOf[DLModel[_]] should be(true)
      dlModel.transform(df).count()
    }
    // TODO find a way to test ML Vectors.
  }

  "An DLEstimator" should "has good result for different batchSize" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val estimator = new DLEstimator[Float](model, criterion, Array(6), Array(1))
      .setBatchSize(51)
      .setMaxEpoch(maxEpoch)
    val data = sc.parallelize(smallData)
    val df: DataFrame = sqlContext.createDataFrame(data).toDF("features", "label")

    val dlModel = estimator.fit(df)
    dlModel.isInstanceOf[DLModel[_]] should be(true)
    dlModel.transform(df).count()
  }

  "An DLEstimator" should "throws exception without correct inputs" in {
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

  "An DLEstimator" should "works in ML pipeline" in {
    var appSparkVersion = org.apache.spark.SPARK_VERSION
    if (appSparkVersion.trim.startsWith("1")) {
      val data = sc.parallelize(
        smallData.map(p => (org.apache.spark.mllib.linalg.Vectors.dense(p._1), p._2)))
      val df: DataFrame = sqlContext.createDataFrame(data).toDF("features", "label")

      val scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaled")
        .setMax(1).setMin(-1)
      val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
      val criterion = ClassNLLCriterion[Float]()
      val estimator = new DLEstimator[Float](model, criterion, Array(6), Array(1))
        .setBatchSize(nRecords)
        // intentionally set low since this only validates data format compatibitliy
        .setMaxEpoch(maxEpoch)
        .setFeaturesCol("scaled")

      val pipeline = new Pipeline().setStages(Array(scaler, estimator))

      val pipelineModel = pipeline.fit(df)
      pipelineModel.isInstanceOf[PipelineModel] should be(true)
      val correct = pipelineModel.transform(df).select("label", "prediction").rdd.filter {
        case Row(label: Double, prediction: Seq[Double]) =>
          label == prediction.indexOf(prediction.max) + 1
      }.count()
      assert(correct > nRecords * 0.8)
    }
  }
}

private case class MinibatchData[T](featureData : Array[T], labelData : Array[T])

object DLEstimatorSpec {
  // Generate noisy input of the form Y = signum(x.dot(weights) + intercept + noise)
  def generateTestInput(
      numRecords: Int,
      weight: Array[Double],
      intercept: Double,
      seed: Long): Seq[(Array[Double], Double)] = {
    val rnd = new Random(seed)
    val data = (1 to numRecords)
      .map( i => Array.tabulate(weight.length)(index => rnd.nextDouble() * 2 - 1))
      .map { record =>
        val y = record.zip(weight).map(t => t._1 * t._2).sum
          +intercept + 0.01 * rnd.nextGaussian()
        val label = if (y > 0) 2.0 else 1.0
        (record, label)
      }
    data
  }
}
