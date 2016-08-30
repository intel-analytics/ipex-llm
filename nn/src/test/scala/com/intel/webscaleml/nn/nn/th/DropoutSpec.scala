package com.intel.webscaleml.nn.nn.th

import com.intel.webscaleml.nn.nn.Dropout
import com.intel.webscaleml.nn.tensor.{Tensor, torch}
import com.intel.webscaleml.nn.th.TH
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import com.intel.webscaleml.nn.tensor.RandomGenerator._

class DropoutSpec extends FlatSpec with BeforeAndAfter with Matchers{
  before{
    if(TH.hasTorch()){
      cancel("Torch is not installed")
    }
  }

  // Todo: Test when input is contiguous
  "Dropout module" should "converate to correct weight and bias" in {
    val module = new Dropout[Double](0.7, false, true)
    val input = torch.Tensor[Double](100,100).t
    val seed = 100

    input.rand()

    val code = "torch.manualSeed(" + seed +")\n" +
      "module = nn.Dropout(0.7)\n" +
      "output1 = module:forward(input)\n" +
      "output2 = module:backward(input, input:clone():fill(1))"

    val (luaTime, torchResult) = TH.run(code, Map("input"->input), Array("output1", "output2"))
    val luaOutput1 = torchResult("output1").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("output2").asInstanceOf[Tensor[Double]]

    val start = System.nanoTime()
    RNG.setSeed(seed)
    val output1 = module.forward(input)
    val output2 = module.backward(input, input.clone().fill(1))
    val end = System.nanoTime()
    val scalaTime = end - start


    luaOutput1 should be (output1)
    luaOutput2 should be (output2)

    println("Test case : Dropout, Torch : " + luaTime + " s, Scala : " + scalaTime/1e9.toDouble + " s")
  }
}
