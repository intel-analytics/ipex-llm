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

import com.intel.analytics.bigdl.nn.Graph
import com.intel.analytics.bigdl.nn.tf._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

class TensorArraySplitSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    "TensorArray serializer Split/Concat" should "work properly" in {
      val tensorArray = new TensorArrayCreator[Float, Float]().inputs()
      val data = Const[Float, Float](Tensor[Float](3, 4).rand()).inputs()
      val lengths = Const[Float, Int](Tensor[Int](T(1, 2))).inputs()
      val splitter = new TensorArraySplit[Float, Float]().inputs((tensorArray, 1), (data, 1),
        (lengths, 1))
      val ctr = new com.intel.analytics.bigdl.nn.tf.ControlDependency[Float]().inputs(splitter)
      val concat = new TensorArrayConcat[Float, Float]().inputs(tensorArray, ctr)
      val size = new TensorArraySize[Float]().inputs(tensorArray, ctr)
      val ctr2 = new com.intel.analytics.bigdl.nn.tf.ControlDependency[Float]().inputs(concat, size)
      val close = new TensorArrayClose[Float]().inputs((tensorArray, 1), (ctr2, 1))
      val model = Graph.dynamic[Float](Array(tensorArray), Array(concat, close, size))

      runSerializationTestWithMultiClass(model, Tensor.scalar[Int](2), Array(
        tensorArray.element.getClass.asInstanceOf[Class[_]],
        splitter.element.getClass.asInstanceOf[Class[_]],
        concat.element.getClass.asInstanceOf[Class[_]],
        close.element.getClass.asInstanceOf[Class[_]],
        size.element.getClass.asInstanceOf[Class[_]]
      ))
    }
  }
}
