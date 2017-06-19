/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

class BackwardSwitchSpec extends FlatSpec with Matchers {

  "BackwardSwitch" should "work properly" in {
    val model = Sequential()
    model.add(Identity())
    model.add(ConcatTable().setName("ct").add(Identity().setName("i1"))
      .add(Identity().setName("i2"))
      .add(BackwardSwitch(Identity()).setName("i3")))
      .add(JoinTable(2, 2))
      .add(SoftMax())

    val input = Tensor[Float](3, 4).rand()

    val label = Tensor[Float](3, 8).range(1, 4)

    val criterion = ClassNLLCriterion()

    model.forward(input)

    criterion.forward(model.output.toTensor[Float], label)

    criterion.backward(model.output.toTensor[Float], label)

    model.backward(input, criterion.gradInput)

    val modules = Utils.getNamedModules(model)

    assert(modules("i3").gradInput == null)

    val expectedConcatGradInput = Tensor[Float].resizeAs(modules("i1").gradInput.toTensor)

    expectedConcatGradInput.add(modules("i1").gradInput.toTensor)
    expectedConcatGradInput.add(modules("i2").gradInput.toTensor)

    modules("ct").gradInput.toTensor.map(expectedConcatGradInput, (a, b) => {
      assert((a - b).abs < 1e-6);
      a
    })
  }


  "Sequential with BackwardSwitch first layer" should "work properly" in {
    val seed = 100
    RNG.setSeed(seed)
    val model = Sequential[Double]()
      .add(BackwardSwitch[Double](Linear[Double](10, 25)).setName("l1"))
      .add(Linear[Double](25, 10).setName("l2"))

    RNG.setSeed(seed)
    val oriModel = Sequential[Double]()
      .add(Linear[Double](10, 25).setName("l1"))
      .add(Linear[Double](25, 10).setName("l2"))

    val input = Tensor[Double](10).randn()
    val gradOutput = Tensor[Double](10).randn()

    model.forward(input)
    oriModel.forward(input)

    model.output should be(oriModel.output)

    val namedModules = Utils.getNamedModules(model)
    val namedModulesOri = Utils.getNamedModules(oriModel)

    model.backward(input, gradOutput)
    oriModel.backward(input, gradOutput)

    namedModules("l2").gradInput.toTensor[Double].map(
      namedModulesOri("l2").gradInput.toTensor[Double], (a, b) => {
        assert((a - b).abs < 1e-6);
        a
      })

    assert(namedModules("l1").gradInput == null)

    model
  }

  "Sequential with BackwardSwitch in the middle" should "work properly" in {
    val seed = 100
    RNG.setSeed(seed)
    val model = Sequential[Double]()
      .add(Identity[Double]().setName("l0"))
      .add(BackwardSwitch[Double](Linear[Double](10, 25)).setName("l1"))
      .add(Linear[Double](25, 10).setName("l2"))

    RNG.setSeed(seed)
    val oriModel = Sequential[Double]()
      .add(Identity[Double]().setName("l0"))
      .add(Linear[Double](10, 25).setName("l1"))
      .add(Linear[Double](25, 10).setName("l2"))

    val input = Tensor[Double](10).randn()
    val gradOutput = Tensor[Double](10).randn()

    model.forward(input)
    oriModel.forward(input)

    model.output should be(oriModel.output)

    val namedModules = Utils.getNamedModules(model)
    val namedModulesOri = Utils.getNamedModules(oriModel)

    model.backward(input, gradOutput)
    oriModel.backward(input, gradOutput)

    namedModules("l2").gradInput.toTensor[Double].map(
      namedModulesOri("l2").gradInput.toTensor[Double], (a, b) => {
        assert((a - b).abs < 1e-6);
        a
      })

    assert(namedModules("l1").gradInput == null)
    assert(namedModules("l0").gradInput == null)
    model
  }

  "Concat with BackwardSwitch" should "work properly" in {
    val seed = 2
    RNG.setSeed(seed)
    val module = new Concat[Double](2)
    val layer1 = Sequential[Double]().setName("l1")
    val layer2 = Sequential[Double]().setName("l2")
    layer1.add(new SpatialBatchNormalization[Double](3, 1e-3))
    layer2.add(new SpatialBatchNormalization[Double](3, 1e-3))
    module.add(layer1).add(layer2)

    val input = Tensor[Double](4, 3, 4, 4).randn()
    val gradOutput = Tensor[Double](4, 6, 4, 4).randn()

    module.forward(input)
    module.backward(input, gradOutput)


    RNG.setSeed(seed)
    val s1 = Sequential[Double]().add(new SpatialBatchNormalization[Double](3, 1e-3))
    val module2 = new Concat[Double](2)
      .add(BackwardSwitch(s1).setName("l1"))
      .add(Sequential[Double]().add(new SpatialBatchNormalization[Double](3, 1e-3)).setName("l2"))

    module2.forward(input)
    module2.backward(input, gradOutput)

    module.output should be(module2.output)

    val namedModules = Utils.getNamedModules(module)
    val namedModules2 = Utils.getNamedModules(module2)

    namedModules("l2").gradInput should be(namedModules2("l2").gradInput)
    assert(namedModules2("l1").gradInput == null)

    namedModules
  }

  "Bottle with BackwardSwitch" should "work properly" in {
    val seed = 100
    RNG.setSeed(seed)
    val module = Bottle[Float](Linear[Float](10, 2), 2, 2)
    module.add(Linear(10, 2))

    RNG.setSeed(seed)
    val module2 = Bottle(BackwardSwitch(Linear[Float](10, 2)), 2, 2)
    module2.add(Linear(10, 2))

    val input = Tensor[Float](4, 5, 10).apply1(_ => Random.nextFloat())
    val gradOutput = Tensor[Float](4, 10).apply1(_ => Random.nextFloat())

    module.forward(input)
    module.backward(input, gradOutput)

    module2.forward(input)
    module2.backward(input, gradOutput)

    module.output should be(module2.output)

    assert(module2.gradInput == null)
  }

  "MapTable with BackwardSwitch" should "work properly" in {
    val input = T(
      Tensor[Float](10).randn(),
      Tensor[Float](10).randn())

    val gradOutput = T(
      Tensor[Float](3).randn(),
      Tensor[Float](3).randn())

    val linear1 = new Linear[Float](10, 3)

    val map = new MapTable[Float]()
    map.add(linear1)
    val map2 = new MapTable[Float]()
    map2.add(BackwardSwitch(linear1.cloneModule()))

    val mapOutput = map.forward(input)
    val mapOutput2 = map2.forward(input)
    mapOutput should equal(mapOutput2)

    val mapGradInput2 = map2.backward(input, gradOutput).toTable

    require(mapGradInput2(1) == null)
    require(mapGradInput2(2) == null)
  }

  "ParallelTable with BackwardSwitch" should "work properly" in {
    val input = T(
      Tensor[Float](10).randn(),
      Tensor[Float](10).randn())

    val gradOutput = T(
      Tensor[Float](3).randn(),
      Tensor[Float](3).randn())

    val linear1 = new Linear[Float](10, 3)
    val linear2 = new Linear[Float](10, 3)
    val expectedOutput = T(
      linear1.updateOutput(input(1)),
      linear2.updateOutput(input(2)))

    val module = new ParallelTable[Float]()
    module.add(linear1)
    module.add(linear2)

    val module2 = ParallelTable[Float]()
      .add(linear1.cloneModule())
      .add(BackwardSwitch(linear2.cloneModule()))
    val output = module.forward(input)
    val output2 = module2.forward(input)

    output should equal(output2)

    module.backward(input, gradOutput)
    module2.backward(input, gradOutput)

    module.gradInput(1).asInstanceOf[Tensor[Float]] should
      equal(module2.gradInput(1).asInstanceOf[Tensor[Float]])

    require(module2.gradInput(2) == null)
  }
}
