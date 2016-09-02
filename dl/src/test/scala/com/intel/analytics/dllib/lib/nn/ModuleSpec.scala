package com.intel.webscaleml.nn.nn

import com.intel.webscaleml.nn.tensor.Storage
import org.scalatest.{Matchers, FlatSpec}

import scala.util.Random

class ModuleSpec extends FlatSpec with Matchers {

  "getParameter" should "behave correctly" in {
    val module = new Sequential[Double]
    val subModule1 = new Linear[Double](2, 3)
    val subModule2 = new Linear[Double](4, 5)
    module.add(subModule1)
    module.add(subModule2)

    val (weight, grad) = module.getParameters()
    weight.dim() should be(1)
    weight.size(1) should be(subModule1.parameters()._1.foldLeft(0)(_ + _.nElement()) +
                               subModule2.parameters()._1.foldLeft(0)(_ + _.nElement()))

    grad.size(1) should be(subModule1.parameters()._2.foldLeft(0)(_ + _.nElement()) +
                               subModule2.parameters()._2.foldLeft(0)(_ + _.nElement()))

    val newValue = Random.nextDouble()
    weight.fill(newValue)
    subModule1.parameters()._1.map( t => t.apply1(e => {e should be(newValue); e}))
    grad.fill(newValue)
    subModule1.parameters()._2.map( t => t.apply1(e => {e should be(newValue); e}))

    val newValue2 = Random.nextDouble()
    weight(Array(1)) = newValue2
    subModule1.parameters()._1(0).storage().asInstanceOf[Storage[Double]].array()(0) should be(newValue2)
  }

  "getParameter from compact tensor" should "not create new storage" in {
    val module = new Sequential[Double]
    val subModule1 = new Linear[Double](2, 3)
    val subModule2 = new Linear[Double](4, 5)
    module.add(subModule1)
    module.add(subModule2)

    val (weight1, grad1) = module.getParameters()
    val (weight2, grad2) = module.getParameters()

    weight1.storage().eq(weight2.storage()) should be(true)
    grad1.storage().eq(grad2.storage()) should be(true)
  }

  "clone module" should "work correctly" in {
    val module = new Sequential[Double]
    module.add(new Linear(2, 3))
    module.add(new Linear(4, 5))

    val module2 = module.cloneModule()

    module2 should be (module)
    module2.eq(module) should be (false)
  }

}
