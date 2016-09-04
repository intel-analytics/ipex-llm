package com.intel.analytics.dllib.lib.nn.th

import com.intel.analytics.dllib.lib.nn.{Linear, MSECriterion}
import com.intel.analytics.dllib.lib.tensor.{Tensor, torch}
import com.intel.analytics.dllib.lib.th.TH
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import com.intel.analytics.dllib.lib.tensor.RandomGenerator._

class LinearSpec extends FlatSpec with BeforeAndAfter with Matchers{
  before{
    if(TH.hasTorch()){
      cancel("Torch is not installed")
    }
  }

  "Linear module" should "converate to correct weight and bias" in {
    val inputN = 5
    val outputN = 2

    val linear = new Linear[Double](inputN, outputN)
    val mse = new MSECriterion[Double]

    val input = torch.Tensor[Double](inputN)
    val res = torch.Tensor[Double](outputN)
    val grad = torch.Tensor[Double](outputN).rand()
    val seed = 100

    input.rand()

    val code = "torch.manualSeed(" + seed +")\n" +
      "linear:reset()\n" +
      "weight = linear.weight\n" +
      "bias = linear.bias\n" +
      "output1 = linear:forward(input)\n" +
      "output2 = linear:backward(input, grad)"

    val (luaTime, torchResult) = TH.run(code, Map("linear"->linear, "input"->input, "grad"->grad), Array("weight", "bias", "output1", "output2"))
    val luaOutput1 = torchResult("output1").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("output2").asInstanceOf[Tensor[Double]]
    val luaWeight = torchResult("weight").asInstanceOf[Tensor[Double]]
    val luaBias = torchResult("bias").asInstanceOf[Tensor[Double]]

    val start = System.nanoTime()
    RNG.setSeed(seed)
    linear.reset()
    val weight = linear.weight
    val bias = linear.bias
    val output1 = linear.forward(input)
    val output2 = linear.backward(input, grad)
    val end = System.nanoTime()
    val scalaTime = end - start

    luaOutput1 should be (output1)
    luaOutput2 should be (output2)
    luaWeight should be (weight)
    luaBias should be (bias)

    println("Test case : Linear, Torch : " + luaTime + " s, Scala : " + scalaTime/1e9.toDouble + " s")
  }

  "Linear (1024, 1000)" should "converate to correct weight and bias" in {
    val inputN = 1024
    val outputN = 1000

    val linear = new Linear[Double](inputN, outputN)
    val mse = new MSECriterion[Double]

    val input = torch.Tensor[Double](inputN).rand()
    val grad = torch.Tensor[Double](outputN).rand()
    val seed = 100

    val code = "torch.manualSeed(" + seed +")\n" +
      "linear:reset()\n" +
      "weight = linear.weight\n" +
      "bias = linear.bias\n" +
      "output1 = linear:forward(input)\n" +
      "output2 = linear:backward(input, grad)"

    val (luaTime, torchResult) = TH.run(code, Map("linear"->linear, "input"->input, "grad"->grad), Array("weight", "bias", "output1", "output2"))
    val luaOutput1 = torchResult("output1").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("output2").asInstanceOf[Tensor[Double]]
    val luaWeight = torchResult("weight").asInstanceOf[Tensor[Double]]
    val luaBias = torchResult("bias").asInstanceOf[Tensor[Double]]

    val start = System.nanoTime()
    RNG.setSeed(seed)
    linear.reset()
    val weight = linear.weight
    val bias = linear.bias
    val output1 = linear.forward(input)
    val output2 = linear.backward(input, grad)
    val end = System.nanoTime()
    val scalaTime = end - start

    luaOutput1 should be (output1)
    luaOutput2 should be (output2)
    luaWeight should be (weight)
    luaBias should be (bias)

    println("Test case : Linear, Torch : " + luaTime + " s, Scala : " + scalaTime/1e9.toDouble + " s")
  }

  "Linear (27, 64)" should "converate to correct weight and bias" in {
    val inputN = 27
    val outputN = 64

    val linear = new Linear[Double](inputN, outputN)

    val input = torch.Tensor[Double](1156, inputN).rand()
    val grad = torch.Tensor[Double](1156, outputN).rand()
    val seed = 100

    val code = "torch.manualSeed(" + seed +")\n" +
      "linear:reset()\n" +
      "weight = linear.weight\n" +
      "bias = linear.bias\n" +
      "output1 = linear:forward(input)\n" +
      "output2 = linear:backward(input, grad)"

    val (luaTime, torchResult) = TH.run(code, Map("linear"->linear, "input"->input, "grad"->grad), Array("weight", "bias", "output1", "output2"))
    val luaOutput1 = torchResult("output1").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("output2").asInstanceOf[Tensor[Double]]
    val luaWeight = torchResult("weight").asInstanceOf[Tensor[Double]]
    val luaBias = torchResult("bias").asInstanceOf[Tensor[Double]]

    val start = System.nanoTime()
    RNG.setSeed(seed)
    linear.reset()
    val weight = linear.weight
    val bias = linear.bias
    val output1 = linear.forward(input)
    val output2 = linear.backward(input, grad)
    val end = System.nanoTime()
    val scalaTime = end - start

    luaOutput1 should be (output1)
    luaOutput2 should be (output2)
    luaWeight should be (weight)
    luaBias should be (bias)

    println("Test case : Linear, Torch : " + luaTime + " s, Scala : " + scalaTime/1e9.toDouble + " s")
  }
}
