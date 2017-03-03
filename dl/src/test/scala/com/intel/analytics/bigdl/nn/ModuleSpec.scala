/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.{RandomGenerator, Table}
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

@com.intel.analytics.bigdl.tags.Parallel
class ModuleSpec extends FlatSpec with Matchers {
  "hashcode()" should "behave correctly" in {
    val r1 = new ReLU[Double]()
    val r2 = new ReLU[Double]()
    val log = new Log[Double]()
    val r3 = new ReLU[Float]()
    val r4 = new ReLU[Double]()
    val r5 = new ReLU[Double]()

    val input1 = Tensor[Double](3, 3).randn()
    r5.forward(input1)
    val input2 = Tensor[Double](3, 3).randn()
    r4.forward(input2)


    r1.hashCode() should equal (r2.hashCode())
    r1.hashCode() should not equal null
    r1.hashCode() should not equal log.hashCode()
    r1.hashCode() should equal (r3.hashCode())
    r5.hashCode() should not equal r4.hashCode()
  }

  "equals()" should "behave correctly" in {
    val r1 = new ReLU[Double]()
    val r2 = new ReLU[Double]()
    val log = new Log[Double]()
    val mNull = null
    val r3 = new ReLU[Float]()
    val r4 = new ReLU[Double]()
    val r5 = new ReLU[Double]()

    val input1 = Tensor[Double](3, 3).randn()
    r5.forward(input1)
    val input2 = Tensor[Double](3, 3).randn()
    r4.forward(input2)


    r1 should equal (r2)
    r1 should not equal null
    r1 should not equal null.asInstanceOf[ReLU[Double]]
    r1 should not equal log
    r1 should not equal mNull
    r1 should equal (r3)
    r5 should not equal r4
  }

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
    subModule1.parameters()._1.map(t => t.apply1(e => {
      e should be(newValue);
      e
    }))
    grad.fill(newValue)
    subModule1.parameters()._2.map(t => t.apply1(e => {
      e should be(newValue);
      e
    }))

    val newValue2 = Random.nextDouble()
    weight(Array(1)) = newValue2
    subModule1.parameters()._1(0).storage().array()(0) should be(newValue2)
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

    module2 should be(module)
    module2.eq(module) should be(false)
  }

  "getName for different module" should "return different name" in {
    val module1 = new Linear[Double](2, 3)
    val module2 = new Linear[Double](4, 5)

    module1.getName() should not be (module2.getName())
  }

  "getName for different module" should "return different name even if the module equals" in {
    val module1 = new Linear[Double](2, 3)
    val module2 = new Linear[Double](2, 3)
    module1.weight.copy(module2.weight)
    module1.bias.copy(module2.bias)

    module1 should be (module2)
    module1.getName() should not be (module2.getName())
  }

  "getParameterTable" should "return right result" in {
    val module = new Sequential[Double]
    val subModule1 = new Linear[Double](2, 3)
    val subModule2 = new Linear[Double](2, 3).setName("fc2")
    module.add(subModule1)
    module.add(subModule2)

    val pt = module.getParametersTable()
    pt[Table](subModule1.getName())[Tensor[Double]]("weight") should be (subModule1.weight)
    pt[Table](subModule1.getName())[Tensor[Double]]("bias") should be (subModule1.bias)
    pt[Table](subModule1.getName())[Tensor[Double]]("gradWeight") should be (subModule1.gradWeight)
    pt[Table](subModule1.getName())[Tensor[Double]]("gradBias") should be (subModule1.gradBias)

    pt[Table](subModule2.getName())[Tensor[Double]]("weight") should be (subModule2.weight)
    pt[Table](subModule2.getName())[Tensor[Double]]("bias") should be (subModule2.bias)
    pt[Table](subModule2.getName())[Tensor[Double]]("gradWeight") should be (subModule2.gradWeight)
    pt[Table](subModule2.getName())[Tensor[Double]]("gradBias") should be (subModule2.gradBias)
  }

}
