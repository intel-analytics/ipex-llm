package com.intel.analytics.dllib.lib.models

import java.util

import com.intel.analytics.dllib.lib.nn.{ClassNLLCriterion, Module, SpatialConvolution}
import com.intel.analytics.dllib.lib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.dllib.lib.tensor.{Tensor, torch}
import scopt.OptionParser


import scala.reflect.ClassTag

/**
  * Performance test for the models
  */
object Perf {
  val parser = new OptionParser[Params]("Performance Test") {
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
      .action((v, p) => p.copy(iteration = v))
    opt[String]('t', "type")
      .text("Data type. It can be float | double")
      .action((v, p) => p.copy(dataType = v))
      .validate(v =>
        if(v.toLowerCase() == "float" || v.toLowerCase() == "double")
          success
        else
          failure("Data type can only be float or double now")
      )
    opt[String]('m', "model")
      .text("Model name. It can be alexnet | alexnetowt | googlenet_v1 | vgg16 | vgg19 | lenet5")
      .action((v, p) => p.copy(module = v))
      .validate(v =>
        if(Set("alexnet", "alexnetowt", "googlenet_v1", "vgg16", "vgg19", "lenet5").contains(v.toLowerCase()))
          success
        else
          failure("Data type can only be alexnet | alexnetowt | googlenet_v1 | vgg16 | vgg19 | lenet5 now")
      )
    help("help").text("Prints this usage text")
  }

  def main(args : Array[String]) : Unit = {
    println("test")
  }
}

case class TestCase[T](input: Tensor[T], target : Tensor[T], model : Module[T])

case class Params(
  batchSize : Int = 128,
  iteration : Int = 10,
  warmUp : Int = 5,
  dataType : String = "float",
  module : String = "alexnet"
)
