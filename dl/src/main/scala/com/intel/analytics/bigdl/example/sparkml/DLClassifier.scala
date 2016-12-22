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
package com.intel.analytics.bigdl.example.sparkml

import java.nio.file.Paths

import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.dataset.image.{GreyImgNormalizer, GreyImgToBatch, SampleToGreyImg}
import com.intel.analytics.bigdl.example.sparkml.MlUtils.{PredictParams, predictParser}
import com.intel.analytics.bigdl.models.lenet.Utils._
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.DLClassifier
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{DataFrame, SQLContext}

import scala.collection.mutable.ArrayBuffer

object DLClassifier {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)

  def main(args: Array[String]): Unit = {
    val batchSize = 10
    predictParser.parse(args, new PredictParams()).map(param => {
      Engine.setCluster(param.nodesNumber, param.coreNumberPerNode)

      val conf = Engine.sparkConf()
        .setAppName("Predict MNIST with trained model")
        .set("spark.akka.frameSize", 64.toString)
        .setMaster("local[1]") // todo
      val sc = new SparkContext(conf)
      val sqlContext = new SQLContext(sc)

      val model = Module.load[Float](param.model)
      val validationData = Paths.get(param.folder, "/t10k-images.idx3-ubyte")
      val validationLabel = Paths.get(param.folder, "/t10k-labels.idx1-ubyte")

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
    })
  }
}
