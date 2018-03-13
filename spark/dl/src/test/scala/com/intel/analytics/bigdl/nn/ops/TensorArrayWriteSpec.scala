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
package com.intel.analytics.bigdl.nn.ops

import com.intel.analytics.bigdl.nn.{Graph, Identity}
import com.intel.analytics.bigdl.nn.tf.Const
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

class TensorArraySerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    "TensorArray serializer R/W" should "work properly" in {
      import com.intel.analytics.bigdl.nn.tf._
      val tensorArray = new TensorArrayCreator[Float, Float]().inputs()
      val data = Const[Float, Float](Tensor.scalar[Float](1)).inputs()
      val index = Const[Float, Int](Tensor.scalar[Int](0)).inputs()
      val write = new TensorArrayWrite[Float, Float]().inputs((tensorArray, 1),
        (index, 1), (data, 1))
      val ctr = new com.intel.analytics.bigdl.nn.tf.ControlDependency[Float]().inputs(write)
      val read = new TensorArrayRead[Float, Float]().inputs((tensorArray, 1), (index, 1), (ctr, 1))
      val grad = new TensorArrayGrad[Float]("grad").inputs(tensorArray)
      val output = Identity[Float]().inputs((grad, 2))
      val model = Graph.dynamic[Float](Array(tensorArray), Array(read, output))

      runSerializationTestWithMultiClass(model, Tensor.scalar[Int](1), Array(
        tensorArray.element.getClass.asInstanceOf[Class[_]],
        write.element.getClass.asInstanceOf[Class[_]],
        read.element.getClass.asInstanceOf[Class[_]],
        grad.element.getClass.asInstanceOf[Class[_]]
      ))
    }
  }
}
