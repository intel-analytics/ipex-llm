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
package com.intel.analytics.bigdl.example.imageclassification

import java.nio.file.Paths

import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.example.imageclassification.MlUtils._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.{DLClassifier => SparkDLClassifier}
import org.apache.spark.sql.SQLContext

/**
 * An example to show how to use DLClassifier Transform
 */
object ImagePredictor {
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

      val model = loadModel(param)
      val partitionNum = param.nodeNumber * param.coreNumber
      val valTrans = new SparkDLClassifier()
        .setInputCol("features")
        .setOutputCol("predict")

      val paramsTrans = ParamMap(
        valTrans.modelTrain -> model,
        valTrans.batchShape ->
        Array(param.batchSize, 3, imageSize, imageSize))

      val valRDD = if (param.isHdfs) {
        // load image set from hdfs
        imagesLoadSeq(param.folder, sc, param.classNum).coalesce(partitionNum, true)
      } else {
        // load image set from local
        val paths = LocalImageFiles.readPaths(Paths.get(param.folder), hasLabel = false)
        sc.parallelize(imagesLoad(paths, 256), partitionNum)
      }

      val transf = RowToByteRecords() ->
          BytesToBGRImg() ->
          BGRImgCropper(imageSize, imageSize) ->
          BGRImgNormalizer(testMean, testStd) ->
          BGRImgToImageVector()

      val valDF = transformDF(sqlContext.createDataFrame(valRDD), transf)

      valTrans.transform(valDF, paramsTrans)
          .select("imageName", "predict")
          .collect()
          .take(param.showNum)
          .foreach(println)
      sc.stop()
    })
  }
}
