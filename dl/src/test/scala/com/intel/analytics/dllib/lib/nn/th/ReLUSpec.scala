package com.intel.webscaleml.nn.nn.th

import com.intel.webscaleml.nn.nn.ReLU
import com.intel.webscaleml.nn.tensor.{Tensor, torch}
import com.intel.webscaleml.nn.th.TH
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.math._
import scala.sys.process._

/**
  * Created by leon on 16-1-18.
  */
class ReLUSpec extends FlatSpec with BeforeAndAfter with Matchers{
  before{
    if(TH.hasTorch()){
      cancel("Torch is not installed")
    }
  }
  
  "A ReLU Module " should "generate correct output and grad" in {
    val module = new ReLU[Double]()
    val input = torch.Tensor[Double](2,2,2)
    input(Array(1,1,1)) = -0.97008799016476
    input(Array(1,1,2)) = -0.89318234380335
    input(Array(1,2,1)) = -0.65073125436902
    input(Array(1,2,2)) = -0.35406025126576
    input(Array(2,1,1)) = -1.0360766677186
    input(Array(2,1,2)) = 1.173689913936
    input(Array(2,2,1)) = 1.6776262558997
    input(Array(2,2,2)) = -0.64814318157732
    val gradOutput = torch.Tensor[Double](2,2,2)
    gradOutput(Array(1,1,1)) = 0.43442418193445
    gradOutput(Array(1,1,2)) = 0.97614445211366
    gradOutput(Array(1,2,1)) = 0.081252868985757
    gradOutput(Array(1,2,2)) = 0.24688877537847
    gradOutput(Array(2,1,1)) = 0.027903598966077
    gradOutput(Array(2,1,2)) = 0.0086153273005038
    gradOutput(Array(2,2,1)) = 0.053113180678338
    gradOutput(Array(2,2,2)) = 0.74842141871341

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input,gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start
    

    val code = "module = nn.ReLU()\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput), Array("output", "gradInput"))
    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    luaOutput1.map(output, (v1, v2) => {
      assert(abs(v1 - v2) == 0);
      v1
    })
    luaOutput2.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) == 0);
      v1
    })

    println("Test case : ReLU, Torch : " + luaTime + " s, Scala : " + scalaTime/1e9.toDouble + " s")
  }

}
