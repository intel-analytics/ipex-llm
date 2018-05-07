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

package com.intel.analytics.bigdl.nn.mkldnn

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.apache.log4j.Logger
import scopt.OptionParser

object Perf {
  val logger: Logger = Logger.getLogger(getClass)

  private def cnnPerf(batchSize: Int, perfIters: Int): Unit = {
//    val model = ResNet_dnn(classNum = 1000, T("depth" -> 50, "optnet" -> true,
//      "dataSet" -> ResNet_dnn.DatasetType.ImageNet))
//    ResNet_dnn.modelInit(model)

    val model = PartialLayers()
    val input = Tensor(batchSize, 3, 224, 224).rand()
    model.forward(input)
    val gradOutput = Tensor().resizeAs(model.output.toTensor).rand()
    model.backward(input, gradOutput)

    val warmIters = 5

    loop(warmIters, ModelWithData(model, input, gradOutput))
    loop(perfIters, ModelWithData(model, input, gradOutput))
  }

  private def doPerf(param: PerfParam): Unit = {
    val batchSize = param.batchSize
    val perfIters = param.perfIters

    cnnPerf(batchSize, perfIters)
  }

  def main(args: Array[String]): Unit = {
    val parser = new OptionParser[PerfParam]("BigDL Local Performance Test") {
      head("Performance Test of Local Optimizer")
      opt[Int]('b', "batchSize")
        .text("Batch size of input data")
        .action((v, p) => p.copy(batchSize = v))
      opt[Int]('i', "iteration")
        .text("Iteration of perf test. The result will be average of each iteration time cost")
        .action((v, p) => p.copy(perfIters = v))
    }
    parser.parse(args, new PerfParam()) match {
      case Some(param) => doPerf(param)
      case None => println("Unknown parameters")
    }
  }

  private def loop(time: Int, modelWithData: ModelWithData): Unit = {
    val model = modelWithData.model
    val input = modelWithData.input
    val gradOutput = modelWithData.gradOutput

    var i = 0
    while (i < time) {
      val start = System.nanoTime()
      model.forward(input)
      model.backward(input, gradOutput)
      val end = System.nanoTime()

      logger.info(
          s"train time ${(end - start) / 1e9}s. " +
          s"Throughput is ${input.size(1).toDouble / (end - start) * 1e9} record / second. "
      )

      i += 1
    }
  }
}

case class ModelWithData(model: Module[Float], input: Tensor[Float], gradOutput: Tensor[Float])

case class PerfParam(
  batchSize: Int = 32,
  perfIters: Int = 100
)
