package com.intel.analytics.dllib.lib.nn.th

import com.intel.analytics.dllib.lib.nn.Tanh
import com.intel.analytics.dllib.lib.tensor.{Tensor, torch}
import com.intel.analytics.dllib.lib.th.TH
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.math._
import scala.sys.process._

/**
  * Created by leon on 16-1-18.
  */
class TanhSpec extends FlatSpec with BeforeAndAfter with Matchers{
  before{
    if(TH.hasTorch()){
      cancel("Torch is not installed")
    }
  }
  
  "A Tanh Module " should "generate correct output and grad" in {
    val module = new Tanh[Double]()
    val input = torch.Tensor[Double](2,2,2)
    input(Array(1,1,1)) = -0.17020166106522
    input(Array(1,1,2)) = 0.57785657607019
    input(Array(1,2,1)) = -1.3404131438583
    input(Array(1,2,2)) = 1.0938102817163
    input(Array(2,1,1)) = 1.120370157063
    input(Array(2,1,2)) = -1.5014141565189
    input(Array(2,2,1)) = 0.3380249235779
    input(Array(2,2,2)) = -0.625677742064
    val gradOutput = torch.Tensor[Double](2,2,2)
    gradOutput(Array(1,1,1)) = 0.79903302760795
    gradOutput(Array(1,1,2)) = 0.019753993256018
    gradOutput(Array(1,2,1)) = 0.63136631483212
    gradOutput(Array(1,2,2)) = 0.29849314852618
    gradOutput(Array(2,1,1)) = 0.94380705454387
    gradOutput(Array(2,1,2)) = 0.030344664584845
    gradOutput(Array(2,2,1)) = 0.33804601291195
    gradOutput(Array(2,2,2)) = 0.8807330634445

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input,gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start
    

    val code = "module = nn.Tanh()\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput), Array("output", "gradInput"))
    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    luaOutput1.map(output, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    luaOutput2.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })

    println("Test case : Tanh, Torch : " + luaTime + " s, Scala : " + scalaTime/1e9.toDouble + " s")
  }
}
