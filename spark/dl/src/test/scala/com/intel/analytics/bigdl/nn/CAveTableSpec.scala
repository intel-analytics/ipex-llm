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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Parallel
class CAveTableSpec extends FlatSpec with Matchers {
  "CAveTable" should "be correct for multiple tensor inputs" in {
    val module = CAveTable[Float]()
    val tensor1 = Tensor[Float](T(1, 2, 3))
    val tensor2 = Tensor[Float](T(2, 3, 4))
    val tensor3 = Tensor[Float](T(3, 4, 5))
    module.forward(T(tensor1, tensor2, tensor3)) should be(Tensor[Float](T(2, 3, 4)))
    val grads = module.backward(T(tensor1, tensor2), Tensor[Float](T(2, 4, 6)))
    grads[Tensor[Float]](1) should be(Tensor[Float](T(1, 2, 3)))
    grads[Tensor[Float]](2) should be(Tensor[Float](T(1, 2, 3)))
  }

}
