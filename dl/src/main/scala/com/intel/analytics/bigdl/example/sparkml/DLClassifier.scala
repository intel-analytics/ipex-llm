/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
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

import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.dataset.image.{BGRImgNormalizer, BGRImgToLabeledPoint, SampleToBGRImg}
import com.intel.analytics.bigdl.example.sparkml.MlUtils._
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.DLClassifier
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SQLContext}
import com.intel.analytics.bigdl.numeric.NumericFloat

/**
 * An example to show how to use DLClassifier Transform, and the test data is cifar10
 */
object DLClassifier {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.bigdl.example").setLevel(Level.INFO)

  def main(args: Array[String]): Unit = {
    predictParser.parse(args, new PredictParams()).map(param => {
      val scc = Engine.init(param.nodeNumber, param.coreNumber, true).map(conf => {
        conf.setAppName("Predict with trained model")
          .set("spark.akka.frameSize", 64.toString)
          .set("spark.task.maxFailures", "1")
        new SparkContext(conf)
      })

      val sc = scc.get
      val sqlContext = new SQLContext(sc)
      val model = Module.load[Float](param.model)

      val valTrans = new DLClassifier()
        .setInputCol("features")
        .setOutputCol("predict")

      val paramsTrans = ParamMap(
        valTrans.modelTrain -> model,
        valTrans.batchShape ->
        Array(param.batchSize, 3, imageSize, imageSize))

      val valSet = DataSet.SeqFileFolder.
        files(param.folder, sc, param.classNum, param.nodeNumber, null) ->
        SampleToBGRImg() ->
        BGRImgNormalizer(testMean, testStd) ->
        BGRImgToLabeledPoint()

      val rowRDD = valSet.data(train = false).asInstanceOf[RDD[LabeledPoint]]
      val testData = sqlContext.createDataFrame(rowRDD)
      valTrans.transform(testData, paramsTrans)
        .select("label", "predict")
        .collect()
        .foreach { case Row(label: Double, predict: Int) =>
          println(label + " " + predict)
        }
      sc.stop()
    })
  }
}
