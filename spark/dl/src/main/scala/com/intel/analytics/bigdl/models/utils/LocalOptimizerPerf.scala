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
package com.intel.analytics.bigdl.models.utils

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{LocalDataSet, MiniBatch}
import com.intel.analytics.bigdl.models.inception.{Inception_v1, Inception_v2}
import com.intel.analytics.bigdl.models.vgg.{Vgg_16, Vgg_19}
import com.intel.analytics.bigdl.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.{Optimizer, Trigger}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import scopt.OptionParser

object LocalOptimizerPerf {
  val parser = new OptionParser[LocalOptimizerPerfParam]("BigDL Local Performance Test") {
    head("Performance Test of Local Optimizer")
    opt[Int]('b', "batchSize")
      .text("Batch size of input data")
      .action((v, p) => p.copy(batchSize = v))
    opt[Int]('c', "coreNumber")
      .text("physical cores number of current machine")
      .action((v, p) => p.copy(coreNumber = v))
    opt[Int]('i', "iteration")
      .text("Iteration of perf test. The result will be average of each iteration time cost")
      .action((v, p) => p.copy(iteration = v))
    opt[String]('m', "model")
      .text("Model name. It can be inception_v1 | vgg16 | vgg19 | " +
        "inception_v2")
      .action((v, p) => p.copy(module = v))
      .validate(v =>
        if (Set("inception_v1", "inception_v2", "vgg16", "vgg19").
          contains(v.toLowerCase())) {
          success
        } else {
          failure("Data type can only be inception_v1 | " +
            "vgg16 | vgg19 | inception_v2 now")
        }
      )
    opt[String]('d', "inputdata")
      .text("Input data type. One of constant | random")
      .action((v, p) => p.copy(inputData = v))
      .validate(v =>
        if (v.toLowerCase() == "constant" || v.toLowerCase() == "random") {
          success
        } else {
          failure("Input data type must be one of constant and random")
        }
      )
    help("help").text("Prints this usage text")
  }

  def main(args: Array[String]): Unit = {
    parser.parse(args, new LocalOptimizerPerfParam()).foreach(performance)
  }

  def performance(param: LocalOptimizerPerfParam): Unit = {
    Engine.setCoreNumber(param.coreNumber)
    val (_model, input) = param.module match {
      case "inception_v1" => (Inception_v1(1000), Tensor(param.batchSize, 3, 224, 224))
      case "inception_v2" => (Inception_v2(1000), Tensor(param.batchSize, 3, 224, 224))
      case "vgg16" => (Vgg_16(1000), Tensor(param.batchSize, 3, 224, 224))
      case "vgg19" => (Vgg_19(1000), Tensor(param.batchSize, 3, 224, 224))
    }
    param.inputData match {
      case "constant" => input.fill(0.01f)
      case "random" => input.rand()
    }
    val model = _model
    println(model)
    val criterion = ClassNLLCriterion()
    val labels = Tensor(param.batchSize).fill(1)
    val dummyDataSet = new LocalDataSet[MiniBatch[Float]] {
      override def data(train : Boolean): Iterator[MiniBatch[Float]] = {
        new Iterator[MiniBatch[Float]] {
          override def hasNext: Boolean = true

          override def next(): MiniBatch[Float] = {
            MiniBatch(input, labels)
          }
        }
      }
      override def size(): Long = 100000
      override def shuffle(): Unit = {}
    }

    Engine.setCoreNumber(param.coreNumber)
    val optimizer = Optimizer(model, dummyDataSet, criterion)
    optimizer.setEndWhen(Trigger.maxIteration(param.iteration)).optimize()
  }
}

/**
 * Local Optimizer Performance Parameters
 *
 * @param batchSize batch size
 * @param coreNumber core number
 * @param iteration how many iterations to run
 * @param dataType data type (double / float)
 * @param module module name
 * @param inputData input data type (constant / random)
 */
case class LocalOptimizerPerfParam(
  batchSize: Int = 128,
  coreNumber: Int = (Runtime.getRuntime().availableProcessors() / 2),
  iteration: Int = 50,
  dataType: String = "float",
  module: String = "inception_v1",
  inputData: String = "random"
)
