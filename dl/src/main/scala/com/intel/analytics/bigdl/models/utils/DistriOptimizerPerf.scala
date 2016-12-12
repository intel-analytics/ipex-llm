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

import com.intel.analytics.bigdl.dataset.{DistributedDataSet, LocalDataSet}
import com.intel.analytics.bigdl.models.imagenet._
import com.intel.analytics.bigdl.models.mnist.LeNet5
import com.intel.analytics.bigdl.nn.{Criterion, ClassNLLCriterion, Module}
import com.intel.analytics.bigdl.optim.{DistriOptimizer, Trigger, LocalOptimizer}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Engine, Activities}
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.rdd.RDD
import scopt.OptionParser

import scala.reflect.ClassTag

object DistriOptimizerPerf {
  val parser = new OptionParser[DistriOptimizerPerfParam]("BigDL Distribute Performance Test") {
    head("Performance Test of Distribute Optimizer")
    opt[Int]('b', "batchSize")
      .text("Batch size of input data")
      .action((v, p) => p.copy(batchSize = v))
    opt[Int]('n', "nodeNumber")
      .text("nodes to run the perf test")
      .action((v, p) => p.copy(nodeNumber = v))
      .required()
    opt[Int]('c', "corePerNode")
      .text("core number of each nodes")
      .action((v, p) => p.copy(corePerNode = v))
      .required()
    opt[Int]('i', "iteration")
      .text("Iteration of perf test. The result will be average of each iteration time cost")
      .action((v, p) => p.copy(iteration = v))
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
    parser.parse(args, new DistriOptimizerPerfParam).map(param => {
      param.dataType match {
        case "float" => performance[Float](param)
        case "double" => performance[Double](param)
        case _ => throw new IllegalArgumentException
      }
    })
  }

  def performance[T: ClassTag](param: DistriOptimizerPerfParam)(implicit tn: TensorNumeric[T]): Unit = {
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

    val conf = Engine.sparkConf().setAppName("DistriOptimizer Performance Test")
    val sc = new SparkContext(conf)
    val broadcast = sc.broadcast((input, labels))
    val rdd = sc.parallelize((1 to param.nodeNumber), param.nodeNumber).mapPartitions(iter => {
      Iterator.single((broadcast.value))
    }).persist()
    rdd.count()
    val dummyDataSet = new DistributedDataSet[(Tensor[T], Tensor[T])] {
      override def size(): Long = 100000
      override def shuffle(): Unit = {}
      override def originRDD(): RDD[_] = rdd
      override def data(): RDD[(Tensor[T], Tensor[T])] = rdd
    }

    val optimizer = new DistriOptimizer[T](
      model,
      dummyDataSet,
      criterion,
      param.nodeNumber,
      param.corePerNode
    )
    optimizer.setEndWhen(Trigger.maxIteration(param.iteration)).optimize()
  }
}

case class DistriOptimizerPerfParam(
  batchSize: Int = 128,
  iteration: Int = 50,
  nodeNumber: Int = -1,
  corePerNode: Int = -1,
  dataType: String = "float",
  module: String = "alexnet",
  inputData: String = "random"
)
