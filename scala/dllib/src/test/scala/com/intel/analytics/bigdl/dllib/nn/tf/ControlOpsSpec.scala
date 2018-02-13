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
package com.intel.analytics.bigdl.nn.tf

import com.intel.analytics.bigdl.nn.ops.Less
import com.intel.analytics.bigdl.nn.{AddConstant, Echo, Graph, Input}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

class ControlOpsSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val input = Input[Float]("input")

    val conditionInput = Input[Float]("conditionInput")
    val const = new com.intel.analytics.bigdl.nn.tf.Const[Float, Float](Tensor(T(9))).inputs()
    val constEnter = new com.intel.analytics.bigdl.nn.tf.Enter[Float]("test_frame").inputs(const)
    val less = Less[Float]().inputs(constEnter, conditionInput)

    val updateInput = Input[Float]()
    val add = AddConstant[Float](1).inputs(updateInput)
    val addEnter = new com.intel.analytics.bigdl.nn.tf.Enter[Float]("test_frame").inputs(add)
    val echo = Echo[Float]().inputs(addEnter)

    val exit = ControlNodes.whileLoop[Float](
      (Seq(conditionInput), less),
      (Seq((updateInput, echo))),
      Seq(input),
      "while"
    )
    val model = Graph.dynamic[Float](Array(input), Array(exit(0)), None, false)
    runSerializationTestWithMultiClass(model, Tensor.scalar[Float](1), Array(
      addEnter.element.getClass.asInstanceOf[Class[_]],
      new com.intel.analytics.bigdl.nn.tf.NextIteration[Float, Float]().getClass,
      new com.intel.analytics.bigdl.nn.tf.Exit[Float]().getClass,
      new com.intel.analytics.bigdl.nn.tf.LoopCondition[Float]().getClass
    ))
  }
}
