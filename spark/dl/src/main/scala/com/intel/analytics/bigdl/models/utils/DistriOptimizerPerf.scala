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
import com.intel.analytics.bigdl.dataset.{DistributedDataSet, MiniBatch}
import com.intel.analytics.bigdl.models.inception.{Inception_v1, Inception_v2}
import com.intel.analytics.bigdl.models.vgg.{Vgg_16, Vgg_19}
import com.intel.analytics.bigdl.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.{Optimizer, Trigger}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import scopt.OptionParser

object DistriOptimizerPerf {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.DEBUG)

  val parser = new OptionParser[DistriOptimizerPerfParam]("BigDL Distribute Performance Test") {
    head("Performance Test of Distribute Optimizer")
    opt[Int]('b', "batchSize")
      .text("Batch size of input data")
      .action((v, p) => p.copy(batchSize = v))
    opt[Int]('e', "maxEpoch")
      .text("epoch numbers of the test")
      .action((v, p) => p.copy(maxEpoch = v))
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
      .text("Model name. It can be inception_v1 | inception_v2 | vgg16 | " +
        "vgg19")
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
    parser.parse(args, new DistriOptimizerPerfParam).foreach(performance)
  }

  def performance(param: DistriOptimizerPerfParam): Unit = {
    val conf = Engine.createSparkConf()
      .setAppName("DistriOptimizer Performance Test")
      .set("spark.task.maxFailures", "1")

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
    val criterion = ClassNLLCriterion[Float]()
    val labels = Tensor(param.batchSize).fill(1)

    val sc = new SparkContext(conf)
    Engine.init
    val broadcast = sc.broadcast(MiniBatch(input, labels))
    val rdd = sc.parallelize((1 to Engine.nodeNumber()), Engine.nodeNumber())
      .mapPartitions(iter => {
        Iterator.single((broadcast.value))
      }).persist()
    rdd.count()
    val dummyDataSet = new DistributedDataSet[MiniBatch[Float]] {
      override def size(): Long = 10000
      override def shuffle(): Unit = {}
      override def originRDD(): RDD[_] = rdd
      override def data(train: Boolean): RDD[MiniBatch[Float]] = rdd
    }

    val optimizer = Optimizer(
      model,
      dummyDataSet,
      criterion
    )
    optimizer.setEndWhen(Trigger.maxEpoch(param.maxEpoch)).optimize()
    sc.stop()
  }
}

/**
 * The parameters of a distributed optimizer
 *
 * @param batchSize batch size
 * @param maxEpoch how many epochs to run
 * @param dataType data type (double / float)
 * @param module module name
 * @param inputData inputData input data type (constant / random)
 */
case class DistriOptimizerPerfParam(
  batchSize: Int = 128,
  maxEpoch: Int = 5,
  dataType: String = "float",
  module: String = "inception_v1",
  inputData: String = "random"
)
