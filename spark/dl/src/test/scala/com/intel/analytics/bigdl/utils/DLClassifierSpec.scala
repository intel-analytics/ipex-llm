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

package com.intel.analytics.bigdl.utils

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random
import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.DLModel
import org.apache.spark.sql.{Row, SQLContext}
import org.scalatest.{FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Parallel
class DLClassifierSpec extends FlatSpec with Matchers {

  "DLClassifier" should "get good result" in {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val conf = new SparkConf().setMaster("local[1]").setAppName("DLClassifier")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    val batchSize = 10
    val model = LeNet5(10)

    // init
    val valTrans = new DLModel[Float](model, Array(28, 28))
      .setFeaturesCol("features")
      .setPredictionCol("predict")
      .setBatchSize(10)

    val tensorBuffer = new ArrayBuffer[Data]()
    // generate test data
    val input = Tensor[Float](10, 28, 28).apply1(e => Random.nextFloat())
    val target = model.forward(input).toTensor[Float]

    val inputArr = input.storage().array()
    val targetArr = target.max(2)._2.squeeze().storage().array()

    (0 until batchSize).foreach(i =>
      tensorBuffer.append(
        Data(targetArr(i), inputArr.slice(i * 28 * 28, (i + 1) * 28 * 28).map(_.toDouble))))

    val rowRDD = sc.parallelize(tensorBuffer)
    val testData = sqlContext.createDataFrame(rowRDD)

    valTrans.transform(testData)
      .select("label", "predict")
      .collect()
      .foreach { case Row(label: Double, predict: mutable.WrappedArray[Double]) =>
        label should be(predict.head)
      }

    tensorBuffer.clear()
    sc.stop()
  }
}

case class Data(label: Double, features: Array[Double])

