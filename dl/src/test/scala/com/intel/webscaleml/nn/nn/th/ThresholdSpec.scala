package com.intel.webscaleml.nn.nn.th

import com.intel.webscaleml.nn.nn.Threshold
import com.intel.webscaleml.nn.tensor.{Tensor, torch}
import com.intel.webscaleml.nn.th.TH
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.math._
import scala.sys.process._

/**
  * Created by leon on 16-1-18.
  */
class ThresholdSpec extends FlatSpec with BeforeAndAfter with Matchers{
  before{
    if(TH.hasTorch()){
      cancel("Torch is not installed")
    }
  }
  
  "A Threshold Module " should "generate correct output and grad" in {
    val module = new Threshold[Double](1,0.8)
    val input = torch.Tensor[Double](2,2,2)
    input(Array(1,1,1)) = -0.89699813351035
    input(Array(1,1,2)) = 1.8529373928905
    input(Array(1,2,1)) = 1.8799053365365
    input(Array(1,2,2)) = 0.076761466450989
    input(Array(2,1,1)) = 1.8863626234233
    input(Array(2,1,2)) = 0.73405137099326
    input(Array(2,2,1)) = 1.3404842875898
    input(Array(2,2,2)) = -0.64910735283047
    val gradOutput = torch.Tensor[Double](2,2,2)
    gradOutput(Array(1,1,1)) = 0.31924905977212
    gradOutput(Array(1,1,2)) = 0.22160539613105
    gradOutput(Array(1,2,1)) = 0.19705923949368
    gradOutput(Array(1,2,2)) = 0.386440459406
    gradOutput(Array(2,1,1)) = 0.12920403806493
    gradOutput(Array(2,1,2)) = 0.7669838971924
    gradOutput(Array(2,2,1)) = 0.10939974407665
    gradOutput(Array(2,2,2)) = 0.70845287665725

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input,gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start
    
    val code = "output = module:forward(input)\n" +
    "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("module" -> module, "input" -> input, "gradOutput" -> gradOutput), Array("output", "gradInput"))
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

    println("Test case : Threshold, Torch : " + luaTime + " s, Scala : " + scalaTime/1e9.toDouble + " s")
  }
}
