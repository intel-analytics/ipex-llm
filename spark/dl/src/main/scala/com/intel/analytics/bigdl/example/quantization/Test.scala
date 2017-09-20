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

package com.intel.analytics.bigdl.example.quantization


import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

object Test {
  import Utils._
  LoggerFilter.redirectSparkInfoLogs()
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)

  def main(args: Array[String]): Unit = {
    testParser.parse(args, TestParams()).foreach { param =>
      val name = param.model
      val path = param.modelPath
      val batchSize = param.batchSize
      val folder = param.folder

      val conf = Engine.createSparkConf()
              .setAppName(s"Test ${name} of ${path} with quantization")
              .set("spark.akka.frameSize", 64.toString)
      val sc = new SparkContext(conf)
      Engine.init

      val partitionNum = Engine.nodeNumber() * Engine.coreNumber()
      val rddData = getRddData(name, sc, partitionNum, folder)
      val transformer = getTransformer(name)
      val evaluationSet = transformer(rddData)

      def loadTF(path: String): Module[Float] = {
        val inputs = Seq("input")
        val outputs = Seq("output")
        Module.loadTF[Float](path, inputs, outputs)
      }

      val loadedModel = if (param.model != "lenet") {
        if (path.endsWith(".pb")) {
          Sequential[Float]()
            .add(ShuffleWithPermutation(Array(1, 3, 4, 2))).add(loadTF(path))
        } else {
          Module.loadModule[Float](path)
        }
      } else {
        val reshape = Reshape[Float](Array(1, 28, 28))
        val newModel = Sequential[Float]()

        if (path.endsWith(".pb")) {
          Sequential[Float]().add(Reshape(Array(1, 28, 28)))
            .add(ShuffleWithPermutation(Array(1, 3, 4, 2))).add(loadTF(path))
        } else {
          Sequential[Float]().add(Reshape(Array(1, 28, 28)))
            .add(Module.loadModule[Float](path))
        }
      }

      val model = if (param.quantize) {
        loadedModel.quantize()
      } else {
        loadedModel
      }

      test(model, evaluationSet, batchSize)

      val (modelResult, modelCosts) = time {
        test(model, evaluationSet, batchSize)
      }

      require(modelResult.length > 0, s"unknown result")
      val totalNum = modelResult(0)._1.result()._2

      val accuracies = new Array[Float](modelResult.length)
      modelResult.indices.foreach { i =>
        accuracies(i) = modelResult(i)._1.result()._1
      }

      val costs = Math.round(totalNum / modelCosts * 100) / 100.0

      writeToLog(param.model, param.quantize, totalNum, accuracies, costs)

      sc.stop()
    }
  }
}
