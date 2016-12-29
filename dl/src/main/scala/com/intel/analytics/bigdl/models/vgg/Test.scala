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

package com.intel.analytics.bigdl.models.vgg

import java.nio.file.Paths

import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.dataset.image.{BGRImgNormalizer, BGRImgToBatch, SampleToBGRImg}
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.optim.{DistriValidator, LocalValidator, Top1Accuracy, Validator}
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

object Test {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)

  import Utils._

  def main(args: Array[String]) {
    testParser.parse(args, new TestParams()).map(param => {
      val sc = Engine.init(param.nodeNumber, param.coreNumber, param.env == "spark")
        .map(conf => {
          conf.setAppName("Test Vgg on Cifar10")
            .set("spark.akka.frameSize", 64.toString)
          new SparkContext(conf)
        })

      val validationSet = (if (sc.isDefined) {
        DataSet.array(Utils.loadTest(param.folder), sc.get)
      } else {
        DataSet.array(Utils.loadTest(param.folder))
      }) -> SampleToBGRImg() -> BGRImgNormalizer(testMean, testStd) ->
        BGRImgToBatch(param.batchSize)

      val model = Module.load[Float](param.model)
      val validator = Validator(model, validationSet)
      val result = validator.test(Array(new Top1Accuracy[Float]))
      result.foreach(r => {
        println(s"${r._2} is ${r._1}")
      })
    })
  }
}


