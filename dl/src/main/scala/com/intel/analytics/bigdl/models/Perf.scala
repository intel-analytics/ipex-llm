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

package com.intel.analytics.bigdl.models

import com.intel.analytics.bigdl.models.imagenet._
import com.intel.analytics.bigdl.models.mnist.LeNet5
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Module}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.Tensor
import scopt.OptionParser

import scala.reflect.ClassTag

/**
 * Performance test for the models
 */
object Perf {
  val parser = new OptionParser[PerfParams]("Performance Test") {
    head("Performance Test of Models")
    opt[Int]('b', "batchSize")
      .text("Batch size of input data")
      .action((v, p) => p.copy(batchSize = v))
    opt[Int]('i', "iteration")
      .text("Iteration of perf test. The result will be average of each iteration time cost")
      .action((v, p) => p.copy(iteration = v))
    opt[Int]('w', "warmUp")
      .text("Warm up iteration number. These iterations will run first and won't be count in " +
        "the perf test result.")
      .action((v, p) => p.copy(warmUp = v))
    opt[String]('t', "type")
      .text("Data type. It can be float | double")
      .action((v, p) => p.copy(dataType = v))
      .validate(v =>
        if (v.toLowerCase() == "float" || v.toLowerCase() == "double") {
          success
        } else {
          failure("Data type can only be float or double now")
        }
      )
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
    opt[String]('e', "engine")
      .text("Engine name. It can be mkl | scala")
      .action((v, p) => p.copy(engine = v))
      .validate(v =>
        if (v.toLowerCase() == "mkl" || v.toLowerCase() == "scala") {
          success
        } else {
          failure("Engine name can only be mkl or scala now")
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
    parser.parse(args, new PerfParams()).map(param => {
      param.dataType match {
        case "float" => performance[Float](param)
        case "double" => performance[Double](param)
        case _ => throw new IllegalArgumentException
      }
    })
  }

  def performance[T: ClassTag](param: PerfParams)(implicit tn: TensorNumeric[T]): Unit = {
    import com.intel.analytics.sparkdl.utils.Engine
    Engine.setCoreNum(2)
    val (model, input) = param.module match {
      case "alexnet" => (AlexNet(1000), Tensor[T](param.batchSize, 3, 227, 227))
      case "alexnetowt" => (AlexNet_OWT(1000), Tensor[T](param.batchSize, 3, 224, 224))
      case "googlenet_v1" => (GoogleNet_v1(1000), Tensor[T](param.batchSize, 3, 224, 224))
      case "googlenet_v2" => (GoogleNet_v2(1000), Tensor[T](param.batchSize, 3, 224, 224))
      case "vgg16" => (Vgg_16(1000), Tensor[T](param.batchSize, 3, 224, 224))
      case "vgg19" => (Vgg_19(1000), Tensor[T](param.batchSize, 3, 224, 224))
      case "lenet5" => (LeNet5(10), Tensor[T](param.batchSize, 1, 28, 28))
    }
    param.inputData match {
      case "constant" => input.fill(tn.fromType(0.01))
      case "random" => input.rand()
    }
    println(model)
    val criterion = new ClassNLLCriterion[T]()
    val labels = Tensor[T](param.batchSize).fill(tn.fromType(1))

    for (i <- 1 to param.warmUp) {
      var time = System.nanoTime()
      val output = model.forward(input)
      criterion.forward(output, labels)
      val forwardTime = System.nanoTime() - time
      time = System.nanoTime()
      val gradOutput = criterion.backward(output, labels)
      model.backward(input, gradOutput)
      val backwardTime = System.nanoTime() - time
      println(s"Warm up iteration $i: forward ${forwardTime / 1e6}ms, " +
        s"backward ${backwardTime / 1e6}ms, " +
        s"total ${(forwardTime + backwardTime) / 1e6}ms")
    }
    model.resetTimes()
    var totalForwardTime = 0L
    var totalBackwardTime = 0L
    for (i <- 1 to param.iteration) {
      var time = System.nanoTime()
      val output = model.forward(input)
      criterion.forward(output, labels)
      val forwardTime = System.nanoTime() - time
      totalForwardTime += forwardTime
      time = System.nanoTime()
      val gradOutput = criterion.backward(output, labels)
      model.backward(input, gradOutput)
      val backwardTime = System.nanoTime() - time
      totalBackwardTime += backwardTime
      println(s"Iteration $i: forward ${forwardTime / 1e6}ms, backward ${backwardTime / 1e6}ms, " +
        s"total ${(forwardTime + backwardTime) / 1e6}ms")
    }
    val times = model.getTimes()
    println("Time cost of each layer(ms):")
    println(times.map(t => (t._1.getName(), (t._2 + t._3) / 1e6 / param.iteration,
      t._2 / 1e6 / param.iteration, t._3 / 1e6 / param.iteration))
      .flatMap(l => Array(s"${l._1}  forward: ${l._3}ms. ", s"${l._1}  backward: ${l._4}ms. "))
      .mkString("\n"))

    println(s"Model is ${param.module}")
    println(s"Average forward time is ${totalForwardTime / param.iteration / 1e6}ms")
    println(s"Average backward time is ${totalBackwardTime / param.iteration / 1e6}ms")
    println(s"Average time is ${(totalForwardTime + totalBackwardTime) / param.iteration / 1e6}ms")

    System.exit(0)
  }
}

case class PerfParams(
  batchSize: Int = 128,
  iteration: Int = 50,
  warmUp: Int = 10,
  dataType: String = "float",
  module: String = "alexnet",
  engine: String = "mkl",
  inputData: String = "random"
)
