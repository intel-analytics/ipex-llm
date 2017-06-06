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
import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor

class PropagateBackSwitchSpec extends FlatSpec with Matchers{
  def getNamedModules[T](model: Module[T]): Map[String, Module[T]] = {
    var namedModules: Map[String, Module[T]] = Map()
    def getModules(module: Module[T]): Unit = {
      module match {
        case m: Container[_, _, T] =>
          namedModules += (module.getName() -> module)
          for (m <- module.asInstanceOf[Container[_, _, T]].modules) getModules(m)
        case _ => namedModules += (module.getName() -> module)
      }
    }
    getModules(model)
    namedModules
  }

  "PropagateBackSwitch" should "work properly" in {
    val model = Sequential()
    model.add(Identity())
    model.add(ConcatTable().setName("ct").add(Identity().setName("i1"))
      .add(Identity().setName("i2"))
      .add(PropagateBackSwitch(Identity()).setName("i3")))
      .add(JoinTable(2, 2))
      .add(SoftMax())

    val input = Tensor[Float](3, 4).rand()

    val label = Tensor[Float](3, 8).range(1, 4)

    val criterion = ClassNLLCriterion()

    model.forward(input)

    criterion.forward(model.output.toTensor[Float], label)

    criterion.backward(model.output.toTensor[Float], label)

    model.backward(input, criterion.gradInput)

    val modules = getNamedModules(model)

    modules("i3").gradInput should be (null)

    val expectedConcatGradInput = Tensor[Float].resizeAs(modules("i1").gradInput.toTensor)

    expectedConcatGradInput.add(modules("i1").gradInput.toTensor)
    expectedConcatGradInput.add(modules("i2").gradInput.toTensor)

    modules("ct").gradInput.toTensor.map(expectedConcatGradInput, (a, b) => {
      assert((a - b).abs < 1e-6); a
    })
  }
}
