package com.intel.analytics.sparkdl.torch

import com.intel.analytics.sparkdl.nn._
import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.utils.RandomGenerator._
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class ClassSimplexCriterionSpec extends FlatSpec with BeforeAndAfter with Matchers {
  before {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }
  }

  "A ClassSimplexCriterion " should "generate correct output and grad" in {
    val seed = 100
    RNG.setSeed(seed)

    val criterion = new ClassSimplexCriterion[Double](6)
    val input = Tensor[Double](1, 6).fill(1)
    val target = Tensor[Double](6).fill(3)

    val start = System.nanoTime()
    val output1 = criterion.forward(input, target)
    val output2 = criterion.backward(input, target)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "criterion = nn.ClassSimplexCriterion(6)\n" +
      "output1 = criterion:forward(input, target)\n " +
      "output2 = criterion:backward(input, target)"


    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "target" -> target),
      Array("output1", "output2"))
    val luaOutput1 = torchResult("output1").asInstanceOf[Double]
    val luaOutput2 = torchResult("output2").asInstanceOf[Tensor[Double]]

  }
}
