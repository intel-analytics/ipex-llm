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
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.SparkContext
import org.apache.spark.ml._
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
    assert(estimator.getMaxEpoch == 100)
    assert(estimator.getBatchSize == 1)
  }

  "An DLClassifier" should "fit on feature(one dimension Array[Double]) and label(Double)" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val classifier = new DLClassifier[Float](model, criterion, Array(6))
      .setBatchSize(nRecords)
      .setMaxEpoch(maxEpoch)
    val data = sc.parallelize(smallData)
    val df = sqlContext.createDataFrame(data).toDF("features", "label")

    val dlModel = classifier.fit(df)
    dlModel.isInstanceOf[DLClassifierModel[_]] should be(true)
    assert(dlModel.transform(df).where("prediction=label").count() > nRecords * 0.8)
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
        // intentionally set low since this only validates data format compatibitliy
        .setMaxEpoch(maxEpoch)
        .setFeaturesCol("scaled")
      val pipeline = new Pipeline().setStages(Array(scaler, estimator))

      val pipelineModel = pipeline.fit(df)
      pipelineModel.isInstanceOf[PipelineModel] should be(true)
      assert(pipelineModel.transform(df).where("prediction=label").count() > nRecords * 0.8)
    }
  }
}

private case class Data(label: Double, features: Array[Double])
