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
package com.intel.analytics.bigdl.tensor

import com.intel.analytics.bigdl.mkl.MklDnn
import com.intel.analytics.bigdl.nn.mkldnn.MemoryOwner
import com.intel.analytics.bigdl.utils.{BigDLSpecHelper, T}
import org.apache.commons.lang3.SerializationUtils
import org.scalatest.BeforeAndAfter

class DnnTensorSpec extends BigDLSpecHelper{
  implicit object Owner extends MemoryOwner {
  }
  "nElement" should "be correct" in {
    val tensor = DnnTensor[Float](3, 4, 5)
    tensor.nElement() should be(3 * 4 * 5)
  }

  "DnnTensor" should "does not support double" in {
    intercept[UnsupportedOperationException] {
      val t = DnnTensor[Double](3, 4, 5)
    }
  }

  "Copy" should "be correct" in {
    val heapTensor = Tensor[Float](T(1, 2, 3, 4))
    val dnnTensor1 = DnnTensor[Float](4)
    dnnTensor1.copy(heapTensor)
    val dnnTensor2 = DnnTensor[Float](4)
    dnnTensor2.copy(dnnTensor1)
    val heapTensor2 = Tensor[Float](4)
    heapTensor2.copy(dnnTensor2)
    heapTensor2 should be(heapTensor)
  }

  "release" should "be correct" in {
    val tensor = DnnTensor[Float](3, 4, 5)
    tensor.isReleased() should be(false)
    tensor.release()
    tensor.isReleased() should be(true)
  }

  "resize" should "be correct" in {
    val tensor = DnnTensor[Float](3, 4)
    tensor.size() should be(Array(3, 4))
    tensor.resize(Array(2, 3))
    tensor.size() should be(Array(2, 3))
    tensor.resize(2)
    tensor.size(1) should be(2)
    tensor.resize(Array(5, 6, 7))
    tensor.size() should be(Array(5, 6, 7))
    tensor.size(2) should be(6)
  }

  "add" should "be correct" in {
    val heapTensor1 = Tensor[Float](T(1, 2, 3, 4))
    val heapTensor2 = Tensor[Float](T(2, 5, 1, 7))
    val dnnTensor1 = DnnTensor[Float](4).copy(heapTensor1)
    val dnnTensor2 = DnnTensor[Float](4).copy(heapTensor2)
    dnnTensor1.add(dnnTensor2)
    val heapTensor3 = Tensor[Float](4).copy(dnnTensor1)
    heapTensor3 should be(Tensor[Float](T(3, 7, 4, 11)))
  }

  "tensor clone with java serialization" should "work correctly" in {
    val heapTensor = Tensor[Float](T(1, 2, 3, 4)).rand(-1, 1)
    val dnnTensor = DnnTensor[Float](4).copy(heapTensor)

    val cloned = SerializationUtils.clone(dnnTensor).asInstanceOf[DnnTensor[Float]]
    val heapCloned = Tensor[Float](4).copy(cloned)

    println(heapTensor)
    println("=" * 80)
    println(heapCloned)

    heapCloned should be (heapTensor)
  }
}
