package com.intel.webscaleml.nn.nn.th

import com.intel.webscaleml.nn.nn.{Linear, Sequential}
import com.intel.webscaleml.nn.tensor.Tensor
import com.intel.webscaleml.nn.th.TH
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.math._
import scala.sys.process._

/**
  * Created by leon on 16-1-18.
  */
class ModuleSpec extends FlatSpec with BeforeAndAfter with Matchers{
  before{
    if(TH.hasTorch()){
      cancel("Torch is not installed")
    }
  }
  
  "getParameter" should "behave correctly" in {
    val module = new Sequential[Double]
    val subModule1 = new Linear[Double](2, 3)
    val subModule2 = new Linear[Double](4, 5)
    module.add(subModule1)
    module.add(subModule2)

    val (weight, grad) = module.getParameters()

    val code = "module = nn.Sequential()\n" +
      "module:add(subModule1)\n" +
      "module:add(subModule2)\n" +
      "weight, grad = module:getParameters()\n"


    val (luaTime, torchResult) = TH.run(code, Map("subModule1"->subModule1, "subModule2"->subModule2), Array("weight", "grad"))
    val luaOutput1 = torchResult("weight").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("grad").asInstanceOf[Tensor[Double]]

    luaOutput1.map(weight, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    luaOutput2.map(grad, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })




  }
}
