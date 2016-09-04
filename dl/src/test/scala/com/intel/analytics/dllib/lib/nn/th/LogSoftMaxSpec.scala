package com.intel.analytics.dllib.lib.nn.th

import com.intel.analytics.dllib.lib.nn.LogSoftMax
import com.intel.analytics.dllib.lib.tensor.{Tensor, torch}
import com.intel.analytics.dllib.lib.th.TH
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.math._
import scala.sys.process._
import scala.util.Random

/**
  * Created by leon on 16-1-18.
  */
class LogSoftMaxSpec extends FlatSpec with BeforeAndAfter with Matchers{
  before{
    if(TH.hasTorch()){
      cancel("Torch is not installed")
    }
  }
  
  "A LogSoftMax Module " should "generate correct output and grad" in {
    val module = new LogSoftMax[Double]()
    Random.setSeed(100)
    val input = torch.Tensor[Double](4, 10).apply1(e => Random.nextDouble())
    val data = torch.Tensor[Double](4, 20).randn()
    val gradOutput = data.narrow(2, 1, 10)

    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start
    
    val code = "module = nn.LogSoftMax()\n" +
      "output1 = module:forward(input)\n " +
      "output2 = module:backward(input, gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput), Array("output1", "output2"))
    val luaOutput = torchResult("output1").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("output2").asInstanceOf[Tensor[Double]]

    luaOutput should be (output)
    luaGradInput should be (gradInput)

    println("Test case : LogSoft, Torch : " + luaTime + " s, Scala : " + scalaTime/1e9.toDouble + " s")
  }
}
