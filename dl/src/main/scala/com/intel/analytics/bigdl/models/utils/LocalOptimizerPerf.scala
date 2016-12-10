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

import com.intel.analytics.bigdl.dataset.LocalDataSet
import com.intel.analytics.bigdl.models.imagenet._
import com.intel.analytics.bigdl.models.mnist.LeNet5
import com.intel.analytics.bigdl.nn.{Module, Criterion, ClassNLLCriterion}
import com.intel.analytics.bigdl.optim.{Trigger, LocalOptimizer}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Activities
import scopt.OptionParser

import scala.reflect.ClassTag

object LocalOptimizerPerf {
  val parser = new OptionParser[LocalOptimizerPerf]("BigDL Local Performance Test") {
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
    parser.parse(args, new LocalOptimizerPerf()).map(param => {
      param.dataType match {
        case "float" => performance[Float](param)
        case "double" => performance[Double](param)
        case _ => throw new IllegalArgumentException
      }
    })
  }

  def performance[T: ClassTag](param: LocalOptimizerPerf)(implicit tn: TensorNumeric[T]): Unit = {
    val (_model, input) = param.module match {
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
    val model = _model.asInstanceOf[Module[Activities, Activities, T]]
    println(model)
    val criterion = ClassNLLCriterion[T]().asInstanceOf[Criterion[Activities, T]]
    val labels = Tensor[T](param.batchSize).fill(tn.fromType(1))
    val dummyDataSet = new LocalDataSet[(Tensor[T], Tensor[T])] {
      override def data(): Iterator[(Tensor[T], Tensor[T])] = {
        new Iterator[(Tensor[T], Tensor[T])] {
          override def hasNext: Boolean = true

          override def next(): (Tensor[T], Tensor[T]) = {
            (input, labels)
          }
        }
      }
      override def size(): Long = 100000
      override def shuffle(): Unit = {}
    }

    val optimizer = new LocalOptimizer[T](model, dummyDataSet, criterion, 1)
    println("Warm up...")
    optimizer.setEndWhen(Trigger.maxIteration(param.warmUp)).optimize()
    println("Warm up done")
    optimizer.setEndWhen(Trigger.maxIteration(param.iteration)).optimize()
  }
}

case class LocalOptimizerPerf(
  batchSize: Int = 128,
  iteration: Int = 50,
  warmUp: Int = 10,
  dataType: String = "float",
  module: String = "alexnet",
  inputData: String = "random"
)
