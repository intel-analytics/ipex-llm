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
package com.intel.analytics.bigdl.models.utils

import com.intel.analytics.bigdl.dataset.{Batch, LocalDataSet}
import com.intel.analytics.bigdl.models.imagenet._
import com.intel.analytics.bigdl.models.vgg.{Vgg_19, Vgg_16}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.optim.{Trigger, LocalOptimizer}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Engine
import scopt.OptionParser

import scala.reflect.ClassTag

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
      .text("Model name. It can be alexnet | alexnetowt | googlenet_v1 | vgg16 | vgg19 | lenet5")
      .action((v, p) => p.copy(module = v))
      .validate(v =>
        if (Set("alexnet", "alexnetowt", "googlenet_v1", "googlenet_v2", "vgg16", "vgg19",
          "lenet5").
          contains(v.toLowerCase())) {
          success
        } else {
          failure("Data type can only be alexnet | alexnetowt | googlenet_v1 | " +
            "vgg16 | vgg19 | lenet5 now")
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
    parser.parse(args, new LocalOptimizerPerfParam()).map(param => {
      performance(param)
    })
  }

  def performance(param: LocalOptimizerPerfParam): Unit = {
    val (_model, input) = param.module match {
      case "alexnet" => (AlexNet(1000), Tensor(param.batchSize, 3, 227, 227))
      case "alexnetowt" => (AlexNet_OWT(1000), Tensor(param.batchSize, 3, 224, 224))
      case "googlenet_v1" => (GoogleNet_v1(1000), Tensor(param.batchSize, 3, 224, 224))
      case "googlenet_v2" => (GoogleNet_v2(1000), Tensor(param.batchSize, 3, 224, 224))
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
    val dummyDataSet = new LocalDataSet[Batch[Float]] {
      override def data(): Iterator[Batch[Float]] = {
        new Iterator[Batch[Float]] {
          override def hasNext: Boolean = true

          override def next(): Batch[Float] = {
            Batch(input, labels)
          }
        }
      }
      override def size(): Long = 100000
      override def shuffle(): Unit = {}
    }

    Engine.setCoreNumber(param.coreNumber)
    val optimizer = new LocalOptimizer[Float](model, dummyDataSet, criterion)
    optimizer.setEndWhen(Trigger.maxIteration(param.iteration)).optimize()
  }
}

case class LocalOptimizerPerfParam(
  batchSize: Int = 128,
  coreNumber: Int = (Runtime.getRuntime().availableProcessors() / 2),
  iteration: Int = 50,
  dataType: String = "float",
  module: String = "alexnet",
  inputData: String = "random"
)
