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

package com.intel.analytics.bigdl.models.resnet

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.models.resnet.Utils._
import com.intel.analytics.bigdl.optim.{Top1Accuracy, ValidationMethod, ValidationResult}
import com.intel.analytics.bigdl.dataset.image.{BGRImgNormalizer, BGRImgToSample, BytesToBGRImg}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

object Test {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)

  def main(args: Array[String]): Unit = {
    testParser.parse(args, TestParams()).foreach { param =>
      val conf = Engine.createSparkConf().setAppName("Test ResNet on Cifar10")
        .set("spark.akka.frameSize", 64.toString)
        .set("spark.task.maxFailures", "1")
      val sc = new SparkContext(conf)

      Engine.init
      val partitionNum = Engine.nodeNumber() * Engine.coreNumber()

      val rddData = sc.parallelize(loadTest(param.folder), partitionNum)
      val transformer = BytesToBGRImg() -> BGRImgNormalizer(Cifar10DataSet.trainMean,
          Cifar10DataSet.trainStd) -> BGRImgToSample()
      val evaluationSet = transformer(rddData)

      val model = Module.load[Float](param.model)
      println(model)
      val result = model.evaluate(evaluationSet, Array(new Top1Accuracy[Float]),
        Some(param.batchSize))
      result.foreach(r => println(s"${r._2} is ${r._1}"))

      sc.stop()
    }
  }
}
