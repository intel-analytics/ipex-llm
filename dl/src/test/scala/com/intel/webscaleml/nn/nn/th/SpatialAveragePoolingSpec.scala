package com.intel.webscaleml.nn.nn.th

import com.intel.webscaleml.nn.nn.SpatialAveragePooling
import com.intel.webscaleml.nn.tensor.{Tensor, torch}
import com.intel.webscaleml.nn.th.TH
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.math._
import scala.sys.process._

/**
  * Created by leon on 16-1-18.
  */
class SpatialAveragePoolingSpec extends FlatSpec with BeforeAndAfter with Matchers{
  before{
    if(TH.hasTorch()){
      cancel("Torch is not installed")
    }
  }
  
  "A SpatialAveragePooling" should "generate correct output and gradInput" in {
    val module = new SpatialAveragePooling[Double](3,2,2,1)
    val input = torch.Tensor[Double](1,4,3)
    input(Array(1,1,1)) = 0.25434372201562
    input(Array(1,1,2)) = 0.20443214406259
    input(Array(1,1,3)) = 0.33442943682894
    input(Array(1,2,1)) = 0.051310112234205
    input(Array(1,2,2)) = 0.56103343307041
    input(Array(1,2,3)) = 0.041837680386379
    input(Array(1,3,1)) = 0.75616162386723
    input(Array(1,3,2)) = 0.35945181339048
    input(Array(1,3,3)) = 0.4502888196148
    input(Array(1,4,1)) = 0.14862711215392
    input(Array(1,4,2)) = 0.050680571002886
    input(Array(1,4,3)) = 0.93014938035049
    val gradOutput = torch.Tensor[Double](1,3,1)
    gradOutput(Array(1,1,1)) = 0.22147525195032
    gradOutput(Array(1,2,1)) = 0.30394183006138
    gradOutput(Array(1,3,1)) = 0.77438542619348

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input,gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start
    


    val code = "module = nn.SpatialAveragePooling(3,2,2,1)\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input"->input, "gradOutput"->gradOutput), Array("output", "gradInput"))
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

    println("Test case : SpatialAveragePooling, Torch : " + luaTime + " s, Scala : " + scalaTime/1e9.toDouble + " s")
  }
}
