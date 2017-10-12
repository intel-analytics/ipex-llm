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
}
