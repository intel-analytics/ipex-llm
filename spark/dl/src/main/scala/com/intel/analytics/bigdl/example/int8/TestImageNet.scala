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

package com.intel.analytics.bigdl.example.int8

import com.intel.analytics.bigdl.models.resnet.ImageNetDataSet
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._
import com.intel.analytics.bigdl.utils._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

/**
 * This example is to evaluate trained resnet50 with imagenet data and get top1 and top5 accuracy
 */
object TestImageNet {
  System.setProperty("bigdl.mkldnn.fusion", "true")

  LoggerFilter.redirectSparkInfoLogs()
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)
  val logger = Logger.getLogger(getClass)

  import Utils._

  def main(args: Array[String]): Unit = {
    testParser.parse(args, new TestParams()).map(param => {
      val conf = Engine.createSparkConf().setAppName("Test model on ImageNet2012")
        .set("spark.rpc.message.maxSize", "200")
      val sc = new SparkContext(conf)
      Engine.init

      val model = if (param.quantize) {
        Module.loadModule[Float](param.model).quantize()
      } else {
        Module.loadModule[Float](param.model)
      }

      val evaluationSet = ImageNetDataSet.valDataSet(param.folder,
        sc, 224, param.batchSize).toDistributed().data(train = false)

      val result = model.evaluate(evaluationSet,
        Array(new Top1Accuracy[Float], new Top5Accuracy[Float]))
      result.foreach(r => println(s"${r._2} is ${r._1}"))

      sc.stop()
    })
  }
}
