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

import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.DLClassifier
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{Row, SQLContext}
import org.scalatest.{FlatSpec, Matchers}

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

@com.intel.analytics.bigdl.tags.Parallel
class DLClassifierSpec extends FlatSpec with Matchers{

  private def processPath(path: String): String = {
    if (path.contains(":")) {
      path.substring(1)
    } else {
      path
    }
  }

  "DLClassifier" should "get good result" in {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val conf = new SparkConf().setMaster("local[1]").setAppName("DLClassifier")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    val batchSize = 10
    val model = LeNet5(10)

    // init
    val valTrans = new DLClassifier[Float]().setInputCol("features").setOutputCol("predict")
    val paramsTrans = ParamMap(valTrans.modelTrain -> model,
      valTrans.batchShape -> Array(10, 28, 28))

    val tensorBuffer = new ArrayBuffer[LabeledPoint]()
    var m = 0
    while (m < 10) {
      // generate test data
      val input = Tensor[Float](10, 28, 28).apply1(e => Random.nextFloat())
      val target = model.forward(input).toTensor[Float]

      val inputArr = input.storage().array()
      val targetArr = target.max(2)._2.squeeze().storage().array()

      var i = 0
      while (i < batchSize) {
        tensorBuffer.append(new LabeledPoint(targetArr(i),
          new DenseVector(inputArr.slice(i * 28 * 28, (i + 1) * 28 * 28).map(_.toDouble))))
        i += 1
      }

      val rowRDD = sc.parallelize(tensorBuffer)
      val testData = sqlContext.createDataFrame(rowRDD)

      valTrans.transform(testData, paramsTrans)
        .select("label", "predict")
        .collect()
        .foreach { case Row(label: Double, predict: Int) =>
          label.toInt should be(predict)
        }

      tensorBuffer.clear()
      m += 1
    }
    sc.stop()
  }
}

