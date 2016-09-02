package com.intel.analytics.dllib.lib.nn.th

import com.intel.analytics.dllib.lib.nn.ClassNLLCriterion
import com.intel.analytics.dllib.lib.tensor.{Tensor, torch}
import com.intel.analytics.dllib.lib.th.TH
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}


/**
  * Created by leon on 16-1-18.
  */
class ClassNLLCriterionSpec extends FlatSpec with BeforeAndAfter with Matchers{
  before{
    if(TH.hasTorch()){
      cancel("Torch is not installed")
    }
  }

  "A ClassNLL Criterion " should "generate correct output and grad" in {
    val criterion = new ClassNLLCriterion[Double]()
    val input = torch.Tensor[Double](3,3)
    input(Array(1,1)) = -1.0262627674932
    input(Array(1,2)) = -1.2412600935171
    input(Array(1,3)) = -1.0423174168648
    input(Array(2,1)) = -0.90330565804228
    input(Array(2,2)) = -1.3686840144413
    input(Array(2,3)) = -1.0778380454479
    input(Array(3,1)) = -0.99131220658219
    input(Array(3,2)) = -1.0559142847536
    input(Array(3,3)) = -1.2692712660404
    val target = torch.Tensor[Double](3)
    target(Array(1)) = 1
    target(Array(2)) = 2
    target(Array(3)) = 3

    val start = System.nanoTime()
    val output1 = criterion.forward(input, target)
    val output2 = criterion.backward(input, target)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "criterion = nn.ClassNLLCriterion()\n" +
      "output1 = criterion:forward(input, target)\n " +
      "output2 = criterion:backward(input, target)"


    val (luaTime, torchResult) = TH.run(code, Map("input"->input, "target"->target), Array("output1", "output2"))
    val luaOutput1 = torchResult("output1").asInstanceOf[Double]
    val luaOutput2 = torchResult("output2").asInstanceOf[Tensor[Double]]

    luaOutput1 should be (output1)
    luaOutput2 should be (output2)
    
    println("Test case : ClassNLLCriterion, Torch : " + luaTime + " s, Scala : " + scalaTime/1e9.toDouble + " s")
  }
}
