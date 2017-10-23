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

import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.numeric.NumericFloat

@com.intel.analytics.bigdl.tags.Parallel
class SparseTensorSpec  extends FlatSpec with Matchers {
  "dim, shape, nElement" should "return right result" in {
    val sTensor1 = Tensor.sparse(Tensor(3, 4).range(1, 12, 1))
    sTensor1.dim() should be (2)
    sTensor1.nElement() should be (12)
    sTensor1.size() should be (Array(3, 4))

    val sTensor2 = Tensor.sparse(Array(Array(1, 2), Array(3, 5)), Array(1f, 2f), Array(3, 5))
    sTensor2.dim() should be (2)
    sTensor2.nElement() should be (2)
    sTensor2.size() should be (Array(3, 5))
  }

  "storageOffset" should "return right result" in {
    val sTensor1 = Tensor.sparse(Tensor(3, 4).range(1, 12, 1))
    sTensor1.storageOffset() should be (1)
  }

  "narrow" should "return right result" in {
    val sTensor = Tensor.sparse(Tensor(6, 5).range(1, 30, 1))
    val sTensor2 = sTensor.narrow(1, 2, 4)
    val sTensor3 = sTensor2.narrow(1, 2, 3)
    sTensor3.storageOffset() should be (11)
    sTensor3.asInstanceOf[SparseTensor[Float]]._indicesOffset should be (Array(2, 0))
  }

  "resize" should "return right result" in {
    val sTensor = Tensor.sparse(Tensor(6, 5).range(1, 30, 1))
    sTensor.resize(Array(10, 10), 50)
    sTensor.size() should be (Array(10, 10))
    sTensor.nElement() should be (50)
    sTensor.storage().array.length should be (50)
  }

  "resize on empty tensor" should "return right result" in {
    val sTensor = Tensor.sparse(Tensor(6, 5).range(1, 30, 1))
    sTensor.set()
    sTensor.resize(Array(10, 10), 50)
    sTensor.size() should be (Array(10, 10))
    sTensor.nElement() should be (50)
    sTensor.storage().array.length should be (50)
  }

  "resize on narrowed tensor" should "return right result" in {
    val sTensor = Tensor.sparse(Tensor(6, 5).range(1, 30, 1)).narrow(1, 2, 4)
    sTensor.resize(Array(10, 10), 50)
    sTensor.size() should be (Array(10, 10))
    sTensor.nElement() should be (50)
    sTensor.storage().array.length should be (55)
    sTensor.storageOffset() should be (6)
  }

  "resize 2D tensor to 3D tensor" should "return right result" in {
    val sTensor = Tensor.sparse(Tensor(6, 5).range(1, 30, 1)).narrow(1, 2, 4)
    sTensor.resize(Array(10, 10, 10), 50)
    sTensor.size() should be (Array(10, 10, 10))
    sTensor.nElement() should be (50)
    sTensor.storage().array.length should be (55)
    sTensor.storageOffset() should be (6)
  }

  "resize 2D tensor to 1D tensor" should "return right result" in {
    val sTensor = Tensor.sparse(Tensor(6, 5).range(1, 30, 1)).narrow(1, 2, 4)
    sTensor.resize(Array(10), 5)
    sTensor.size() should be (Array(10))
    sTensor.nElement() should be (5)
    sTensor.storage().array.length should be (30)
    sTensor.storageOffset() should be (6)
  }

  "resize 2D tensor to 1D tensor" should "return right result2" in {
    val sTensor = Tensor.sparse(Tensor(6, 5).range(1, 30, 1))
    sTensor.resize(Array(30), 30)
    sTensor.size() should be (Array(30))
    sTensor.nElement() should be (30)
    sTensor.storage().array.length should be (30)
    sTensor.storageOffset() should be (1)
  }

}
