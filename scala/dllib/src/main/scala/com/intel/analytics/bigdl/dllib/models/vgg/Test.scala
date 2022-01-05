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

package com.intel.analytics.bigdl.dllib.models.vgg

import com.intel.analytics.bigdl.dllib.feature.dataset.DataSet
import com.intel.analytics.bigdl.dllib.feature.dataset.image._
import com.intel.analytics.bigdl.dllib.models.lenet.Utils._
import com.intel.analytics.bigdl.dllib.nn.Module
import com.intel.analytics.bigdl.dllib.optim.{Top1Accuracy, Validator}
import com.intel.analytics.bigdl.dllib.utils.Engine
import org.apache.logging.log4j.Level
import org.apache.logging.log4j.core.config.Configurator
import org.apache.spark.SparkContext

object Test {
  Configurator.setLevel("org", Level.ERROR)
  Configurator.setLevel("akka", Level.ERROR)
  Configurator.setLevel("breeze", Level.ERROR)


  import Utils._

  def main(args: Array[String]) {
    testParser.parse(args, new TestParams()).foreach { param =>
      val conf = Engine.createSparkConf().setAppName("Test Vgg on Cifar10")
        .set("spark.akka.frameSize", 64.toString)
      val sc = new SparkContext(conf)
      Engine.init

      val partitionNum = Engine.nodeNumber() * Engine.coreNumber()
      val rddData = sc.parallelize(Utils.loadTest(param.folder), partitionNum)
      val transformer = BytesToBGRImg() -> BGRImgNormalizer(testMean, testStd) -> BGRImgToSample()
      val evaluationSet = transformer(rddData)

      val model = Module.loadModule[Float](param.model)
      val result = model.evaluate(evaluationSet,
        Array(new Top1Accuracy[Float]), Some(param.batchSize))
      result.foreach(r => println(s"${r._2} is ${r._1}"))
      sc.stop()
    }
  }
}


