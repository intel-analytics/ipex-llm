package com.intel.webscaleml.nn.nn.th

import com.intel.webscaleml.nn.nn.SpatialMaxPooling
import com.intel.webscaleml.nn.tensor.{Tensor, torch}
import com.intel.webscaleml.nn.th.TH
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.math._
import scala.sys.process._

/**
  * Created by leon on 16-1-18.
  */
class SpatialMaxPoolingSpec extends FlatSpec with BeforeAndAfter with Matchers{
  before{
    if(TH.hasTorch()){
      cancel("Torch is not installed")
    }
  }
  
  "A SpatialMaxPooling" should "generate correct output and gradInput" in {
    val module = new SpatialMaxPooling[Double](2, 2)
    val input = torch.Tensor[Double](1, 3, 3)
    input(Array(1, 1, 1)) = 0.53367262030952
    input(Array(1, 1, 2)) = 0.79637692729011
    input(Array(1, 1, 3)) = 0.56747663160786
    input(Array(1, 2, 1)) = 0.18039962812327
    input(Array(1, 2, 2)) = 0.24608615692705
    input(Array(1, 2, 3)) = 0.22956256521866
    input(Array(1, 3, 1)) = 0.30736334621906
    input(Array(1, 3, 2)) = 0.59734606579877
    input(Array(1, 3, 3)) = 0.42989541869611
    val gradOutput = torch.Tensor[Double](1, 1, 1)
    gradOutput(Array(1, 1, 1)) = 0.023921491578221

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("module" -> module, "input" -> input, "gradOutput" -> gradOutput), Array("output", "gradInput"))
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

    println("Test case : SpatialMaxPooling, Torch : " + luaTime + " s, Scala : " + scalaTime/1e9.toDouble + " s")
  }
}
