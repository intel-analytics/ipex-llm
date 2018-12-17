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

import com.intel.analytics.bigdl.nn.Graph
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

class StackOpsSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val data = Const[Float, Float](Tensor.scalar[Float](1)).inputs()
    val stack = new StackCreator[Float, Float]().inputs()
    val push = new StackPush[Float, Float]().inputs(stack, data)
    val ctr = new com.intel.analytics.bigdl.nn.tf.ControlDependency[Float]().inputs(push)
    val pop = new StackPop[Float, Float]().inputs(stack, ctr)
    val model = Graph.dynamic[Float](Array(stack), Array(pop))

    runSerializationTestWithMultiClass(model, Tensor.scalar(1), Array(
      stack.element.getClass.asInstanceOf[Class[_]],
      push.element.getClass.asInstanceOf[Class[_]],
      pop.element.getClass.asInstanceOf[Class[_]]
    ))
  }
}
