package com.intel.webscaleml.nn.nn.th

import com.intel.webscaleml.nn.nn.Reshape
import com.intel.webscaleml.nn.tensor.{Tensor, torch}
import com.intel.webscaleml.nn.th.TH
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.math._
import scala.sys.process._

/**
  * Created by leon on 16-1-18.
  */
class ReshapeSpec extends FlatSpec with BeforeAndAfter with Matchers{
  before{
    if(TH.hasTorch()){
      cancel("Torch is not installed")
    }
  }

  "A Reshape Module " should "generate correct output and grad" in {
    val module = new Reshape[Double](Array(3,2))

    val input = torch.Tensor[Double](4,1,6)
    input.rand()
    val gradOutput = torch.Tensor[Double](4,3,2)
    gradOutput.rand()

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input,gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start
    

    val code = "module = nn.Reshape(3,2)\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input"->input, "gradOutput"->gradOutput), Array("output", "gradInput"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]

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

    println("Test case : Reshape, Torch : " + luaTime + " s, Scala : " + scalaTime/1e9.toDouble + " s")
  }
}
