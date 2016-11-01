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

package com.intel.analytics.sparkdl.models

import java.util.concurrent.Executors

import com.github.fommil.netlib.{BLAS, NativeSystemBLAS}
import com.intel.analytics.sparkdl.utils.T
import com.intel.analytics.sparkdl.models.imagenet.ResNet.ShortcutType
import com.intel.analytics.sparkdl.models.imagenet._
import com.intel.analytics.sparkdl.nn.{ClassNLLCriterion, CrossEntropyCriterion, Module}
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.tensor.Tensor
import scopt.OptionParser

import scala.concurrent.{Await, ExecutionContext, Future}
import scala.concurrent.duration.Duration
import scala.reflect.ClassTag

/**
 * Performance test for the models, in this program, we rum multiple models, each model train
 * a small batch. This is better for some complex model(e.g googlenet) compare to single model
 * train with a large batch
 */
object MultiModelPerf {
  val parser = new OptionParser[MultiModelPerfParams]("Performance Test") {
    head("Performance Test of Models")
    opt[Int]('b', "batchSize")
      .text("Batch size of input data")
      .action((v, p) => p.copy(batchSize = v))
    opt[Int]('i', "iteration")
      .text("Iteration of perf test. The result will be average of each iteration time cost")
      .action((v, p) => p.copy(iteration = v))
    opt[Int]('c', "cores")
      .text("Used cores")
      .action((v, p) => p.copy(cores = v))
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
      .text("Model name. It can be alexnet | alexnetowt | googlenet_v1 | googlenet_v2")
      .action((v, p) => p.copy(module = v))
      .validate(v =>
        if (Set("alexnet", "alexnetowt", "googlenet_v1", "googlenet_v2", "resnet").
          contains(v.toLowerCase())) {
          success
        } else {
          failure("Data type can only be alexnet | alexnetowt | googlenet_v1 | " +
            "vgg16 | vgg19 | lenet5 now")
        }
      )
    help("help").text("Prints this usage text")
  }

  def main(args: Array[String]): Unit = {
    parser.parse(args, new MultiModelPerfParams()).map(param => {
      param.dataType match {
        case "float" => performance[Float](param)
        case "double" => performance[Double](param)
        case _ => throw new IllegalArgumentException
      }
    })
  }

  def performance[T: ClassTag](param: MultiModelPerfParams)(implicit tn: TensorNumeric[T]): Unit = {
    val tests = (1 to param.cores).map(_ => param.module match {
      case "alexnet" => (AlexNet(1000), Tensor[T](param.batchSize, 3, 227, 227).rand(),
        new ClassNLLCriterion[T](), Tensor[T](param.batchSize).fill(tn.fromType(1)))
      case "alexnetowt" => (AlexNet_OWT(1000), Tensor[T](param.batchSize, 3, 224, 224).rand(),
        new ClassNLLCriterion[T](), Tensor[T](param.batchSize).fill(tn.fromType(1)))
      case "googlenet_v1" => (GoogleNet_v1(1000), Tensor[T](param.batchSize, 3, 224, 224).rand(),
        new ClassNLLCriterion[T](), Tensor[T](param.batchSize).fill(tn.fromType(1)))
      case "googlenet_v2" => (GoogleNet_v2(1000), Tensor[T](param.batchSize, 3, 224, 224).rand(),
        new ClassNLLCriterion[T](), Tensor[T](param.batchSize).fill(tn.fromType(1)))
      case "resnet" => (ResNet(1000, T("shortcutType" -> ShortcutType.B, "depth"->50)), Tensor[T](param.batchSize, 3, 224, 224).rand(),
        new CrossEntropyCriterion[T](), Tensor[T](param.batchSize).fill(tn.fromType(1)))
    })
    require(BLAS.getInstance().isInstanceOf[NativeSystemBLAS])
    if (param.module == "resnet") {
      tests.map(x => {ResNet.shareGradInput(x._1)
                      ResNet.convInit("SpatialConvolution", x._1)
                      ResNet.bnInit("SpatialBatchNormalization", x._1)})
    }

    val grads = tests.map(_._1.getParameters()._2).toArray
    val gradLength = grads(0).nElement()
    val taskSize = gradLength / param.cores
    val extraTask = gradLength % param.cores

    implicit val context = new ExecutionContext {
      val threadPool = Executors.newFixedThreadPool(param.cores)

      def execute(runnable: Runnable) {
        threadPool.submit(runnable)
      }

      def reportFailure(t: Throwable) {}
    }

    for (i <- 1 to param.warmUp) {
      val time = System.nanoTime()
      (0 until param.cores).map(j => Future {
        val (model, input, criterion, labels) = tests(j)
        val output = model.forward(input)
        criterion.forward(output, labels)
        val gradOutput = criterion.backward(output, labels)
        model.backward(input, gradOutput)
      }).foreach(Await.result(_, Duration.Inf))

      (0 until param.cores).map(tid => Future {
        val offset = tid * taskSize + math.min(tid, extraTask)
        val length = taskSize + (if (tid < extraTask) 1 else 0)
        var i = 1
        while (i < grads.length) {
          grads(0).narrow(1, offset + 1, length).add(grads(i).narrow(1, offset + 1, length))
          i += 1
        }
      }).foreach(Await.result(_, Duration.Inf))

      val total = System.nanoTime() - time
      println(s"Warmup Iteration $i: total ${total / 1e6}ms")
    }
    tests.foreach(_._1.resetTimes())

    var totalTime = 0L
    for (i <- 1 to param.iteration) {
      val time = System.nanoTime()
      (0 until param.cores).map(j => Future {
        val (model, input, criterion, labels) = tests(j)
        val output = model.forward(input)
        criterion.forward(output, labels)
        val gradOutput = criterion.backward(output, labels)
        model.backward(input, gradOutput)
      }).foreach(Await.result(_, Duration.Inf))

      (0 until param.cores).map(tid => Future {
        val offset = tid * taskSize + math.min(tid, extraTask)
        val length = taskSize + (if (tid < extraTask) 1 else 0)
        var i = 1
        while (i < grads.length) {
          grads(0).narrow(1, offset + 1, length).add(grads(i).narrow(1, offset + 1, length))
          i += 1
        }
      }).foreach(Await.result(_, Duration.Inf))
      val total = System.nanoTime() - time
      totalTime += total
      println(s"Iteration $i: total ${total / 1e6}ms")
    }
    println(s"Total average time ${totalTime / 1e6 / param.iteration}ms")

    System.exit(0)
  }
}

case class MultiModelPerfParams(
  batchSize: Int = 50,
  iteration: Int = 20,
  cores: Int = 25,
  warmUp: Int = 10,
  dataType: String = "float",
  module: String = "resnet"
)
