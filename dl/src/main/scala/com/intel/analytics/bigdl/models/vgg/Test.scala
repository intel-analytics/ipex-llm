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

package com.intel.analytics.bigdl.models.vgg

import java.nio.file.Paths
import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.dataset.image.{RGBImgToBatch, RGBImgNormalizer}
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.optim.{DistriValidator, Top1Accuracy, LocalValidator}
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

object LocalTest {

  import Utils._

  def main(args: Array[String]) {
    val batchSize = 128
    testLocalParser.parse(args, new TestLocalParams()).map(param => {
      Engine.setCoreNumber(param.coreNumber)
      val validationSet = DataSet.ImageFolder
        .images(Paths.get(param.folder), 32)
        .transform(RGBImgNormalizer(testMean, testStd))
        .transform(RGBImgToBatch(batchSize))

      val model = Module.load[Float](param.model)
      val validator = new LocalValidator[Float](model)
      val result = validator.test(validationSet, Array(new Top1Accuracy[Float]))
      result.foreach(r => {
        println(s"${r._2} is ${r._1}")
      })
    })
  }
}

object SparkTest {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)

  import Utils._

  def main(args: Array[String]) {
    val batchSize = 128
    testSparkParser.parse(args, new TestSparkParams()).map(param => {
      Engine.setCluster(param.nodesNumber, param.coreNumberPerNode)

      val conf = Engine.sparkConf()
        .setAppName("Test Vgg on Cifar10")
        .set("spark.akka.frameSize", 64.toString)
      val sc = new SparkContext(conf)
      val validationSet = DataSet.ImageFolder
        .images(Paths.get(param.folder), sc, param.nodesNumber, 32)
        .transform(RGBImgNormalizer(testMean, testStd))
        .transform(RGBImgToBatch(batchSize))

      val model = Module.load[Float](param.model)
      val validator = new DistriValidator[Float](model)
      val result = validator.test(validationSet, Array(new Top1Accuracy[Float]))
      result.foreach(r => {
        println(s"${r._2} is ${r._1}")
      })
    })
  }
}


