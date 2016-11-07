package com.intel.analytics.sparkdl.torch

import com.intel.analytics.sparkdl.nn.CosineDistance
import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.utils.RandomGenerator._
import com.intel.analytics.sparkdl.utils.Table
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.collection.mutable.HashMap
import scala.util.Random

class CosineDistanceSpec extends FlatSpec with BeforeAndAfter with Matchers{
  before {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }
  }

  "A CosineDistance " should "generate correct output and grad" in {
    val seed = 100
    RNG.setSeed(seed)

    val input1 = Tensor[Double](3).apply1(e => Random.nextDouble())
    val input2 = Tensor[Double](3).apply1(e => Random.nextDouble())
    val gradOutput = Tensor[Double](1).apply1(e => Random.nextDouble())

    var input = new Table()
    input(1.toDouble) = input1
    input(2.toDouble) = input2

    val code = "torch.manualSeed(" + seed + ")\n" +
      "module = nn.CosineDistance()\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)\n"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))

    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[HashMap[Double, Tensor[Double]]]

    val module = new CosineDistance[Double]()
    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    output should be(luaOutput1)

    val luagradInput1 = luaOutput2.get(1.0).getOrElse(null)
    val luagradInput2 = luaOutput2.get(2.0).getOrElse(null)

    val gradInput1 = gradInput.apply(1.toDouble).asInstanceOf[Tensor[Double]]
    val gradInput2 = gradInput.apply(2.toDouble).asInstanceOf[Tensor[Double]]
    gradInput1 should be(luagradInput1)
    gradInput2 should be(luagradInput2)

    println("Test case : Bilinear, Torch : " + luaTime +
      " s, Scala : " + scalaTime / 1e9 + " s")
  }
}
