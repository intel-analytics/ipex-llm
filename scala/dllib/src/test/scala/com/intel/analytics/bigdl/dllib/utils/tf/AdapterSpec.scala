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
package com.intel.analytics.bigdl.utils.tf

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.{Module, Reshape}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{BigDLSpecHelper, T}
import com.intel.analytics.bigdl.utils.tf.loaders.Adapter
import org.scalatest.{FlatSpec, Matchers}

class AdapterSpec extends BigDLSpecHelper {

  private val module = new AdapterForTest()

  "Adapter"  should "work correct" in {
    module.forward(T(Tensor[Float](3, 4), Tensor[Int](T(4, 3)))) should be(Tensor[Float](4, 3))
    module.forward(T(Tensor[Float](3, 4), Tensor[Int](T(4, 3)))) should be(Tensor[Float](4, 3))
  }

  "Adapter"  should "rebuild module when const tensor is changed" in {
    module.forward(T(Tensor[Float](3, 4), Tensor[Int](T(4, 3)))) should be(Tensor[Float](4, 3))
    module.forward(T(Tensor[Float](3, 4), Tensor[Int](T(2, 6)))) should be(Tensor[Float](2, 6))
  }

  "Adapter" should "be able to serialized and deserialized" in {
    val tmpFile = createTmpFile()
    tmpFile.delete()
    module.saveModule(tmpFile.getAbsolutePath)
    val loadModule = Module.loadModule[Float](tmpFile.getAbsolutePath)
    loadModule.forward(T(Tensor[Float](3, 4), Tensor[Int](T(4, 3)))) should be(Tensor[Float](4, 3))
  }
}

class AdapterForTest extends Adapter[Float](Array(2)) {
  override def build(tensorArrays: Array[Tensor[_]])
  : AbstractModule[Activity, Activity, Float] = {
    val sizes = tensorArrays(0).asInstanceOf[Tensor[Int]]

    val batchMode = sizes.valueAt(1) == -1
    val arraySize = new Array[Int](if (batchMode) sizes.nElement() - 1 else sizes.nElement())
    var i = if (batchMode) 2 else 1
    var k = 0
    while(i <= sizes.nElement()) {
      arraySize(k) = sizes.valueAt(i)
      k += 1
      i += 1
    }
    Reshape[Float](size = arraySize, Some(batchMode))
  }
}
