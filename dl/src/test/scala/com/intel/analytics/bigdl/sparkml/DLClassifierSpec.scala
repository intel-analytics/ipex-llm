/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.sparkml

import java.io.File
import java.nio.file.Paths

import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.dataset.image.{GreyImgNormalizer, GreyImgToBatch, SampleToGreyImg}
import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.models.lenet.Utils._
import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.DLClassifier
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.scalatest.{FlatSpec, Matchers}

import scala.collection.mutable.ArrayBuffer

class DLClassifierSpec extends FlatSpec with Matchers{

  private def processPath(path: String): String = {
    if (path.contains(":")) {
      path.substring(1)
    } else {
      path
    }
  }

  case class Person(orgLabel: Float, features: Tensor[Float])

  "DLClassifier for MNIST" should "get good result" in {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val sc = new SparkContext("local[1]", "DLClassifier")
    val sqlContext = new SQLContext(sc)
    val batchSize = 10
    val model = LeNet5(10)

    // read test data
    val resource = getClass().getClassLoader().getResource("mnist")
    val validationData = Paths.get(processPath(resource.getPath()) + File.separator,
      "t10k-images.idx3-ubyte")
    val validationLabel = Paths.get(processPath(resource.getPath()) + File.separator,
      "t10k-labels.idx1-ubyte")

    val validationSet = DataSet.array(load(validationData, validationLabel))
      .transform(SampleToGreyImg(28, 28))
    val normalizerVal = GreyImgNormalizer(validationSet)
    val valSet = validationSet.transform(normalizerVal)
      .transform(GreyImgToBatch(batchSize))
    val valData = valSet.data(looped = false)

    // init
    val valTrans = new DLClassifier().setInputCol("features").setOutputCol("predict")
    val paramsTrans = ParamMap(valTrans.modelTrain -> model,
      valTrans.batchSize -> Array(10, 28, 28))
    val tensorBuffer = new ArrayBuffer[LabeledPoint]()
    var res: DataFrame = null

    while (valData.hasNext) {
      val batch = valData.next()
      val input = batch.data.storage().array()
      val target = batch.labels.storage().array()

      var i = 0
      while (i < batchSize) {
        tensorBuffer.append(new LabeledPoint(target(i),
          new DenseVector(input.slice(i * 28 * 28, (i + 1 ) * 28 * 28).map(_.toDouble))))
        i += 1
      }

      val rowRDD = sc.parallelize(tensorBuffer)
      val testData = sqlContext.createDataFrame(rowRDD)
      res = valTrans.transform(testData, paramsTrans)
      res.select("label", "predict").show()

      tensorBuffer.clear()
    }
    sc.stop()
  }
}

