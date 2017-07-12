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

package com.intel.analytics.bigdl.dataset

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{FlatSpec, Matchers}

class MiniBatchSpec extends FlatSpec with Matchers {
  "TensorMiniBatch size" should "return right result" in {
    val a = Tensor[Float](3, 4).range(1, 12, 1)
    val b = Tensor[Float](3).range(1, 3, 1)
    val miniBatch = MiniBatch(a, b)
    miniBatch.size() should be (3)
  }

  "TensorMiniBatch getInput/target" should "return right result" in {
    val a = Tensor[Float](3, 4).range(1, 12, 1)
    val b = Tensor[Float](3).range(1, 3, 1)
    val miniBatch = MiniBatch(a, b)
    miniBatch.getInput() should be (a)
    miniBatch.getTarget() should be (b)
  }

  "TensorMiniBatch slice" should "return right result" in {
    val a = Tensor[Float](3, 4).range(1, 12, 1)
    val b = Tensor[Float](3).range(1, 3, 1)
    val miniBatch = MiniBatch(a, b)

    miniBatch.slice(1, 1).getInput() should be (Tensor[Float](1, 4).range(1, 4, 1))
    miniBatch.slice(2, 1).getInput() should be (Tensor[Float](1, 4).range(5, 8, 1))
    miniBatch.slice(3, 1).getInput() should be (Tensor[Float](1, 4).range(9, 12, 1))
    miniBatch.slice(1, 1).getTarget() should be (Tensor[Float](1).fill(1))
    miniBatch.slice(2, 1).getTarget() should be (Tensor[Float](1).fill(2))
    miniBatch.slice(3, 1).getTarget() should be (Tensor[Float](1).fill(3))
  }

  "ArrayTensorMiniBatch size" should "return right result" in {
    val a1 = Tensor[Float](3, 4).range(1, 12, 1)
    val a2 = Tensor[Float](3, 2).range(1, 6, 1)
    val b = Tensor[Float](3).range(1, 3, 1)
    val miniBatch = MiniBatch(Array(a1, a2), b)
    miniBatch.size() should be (3)
  }

  "ArrayTensorMiniBatch getInput/target" should "return right result" in {
    val a1 = Tensor[Float](3, 4).range(1, 12, 1)
    val a2 = Tensor[Float](3, 2).range(1, 6, 1)
    val b = Tensor[Float](3).range(1, 3, 1)
    val miniBatch = MiniBatch(Array(a1, a2), b)
    miniBatch.getInput() should be (T(a1, a2))
    miniBatch.getTarget() should be (b)
  }

  "ArrayTensorMiniBatch slice" should "return right result" in {
    val a1 = Tensor[Float](3, 2, 2).range(1, 12, 1)
    val a2 = Tensor[Float](3, 2).range(1, 6, 1)
    val b = Tensor[Float](3).range(1, 3, 1)
    val miniBatch = MiniBatch(Array(a1, a2), b)

    miniBatch.slice(1, 1).getInput() should be (T(Tensor[Float](1, 2, 2).range(1, 4, 1),
      Tensor[Float](1, 2).range(1, 2, 1)))
    miniBatch.slice(2, 1).getInput() should be (T(Tensor[Float](1, 2, 2).range(5, 8, 1),
      Tensor[Float](1, 2).range(3, 4, 1)))
    miniBatch.slice(3, 1).getInput() should be (T(Tensor[Float](1, 2, 2).range(9, 12, 1),
      Tensor[Float](1, 2).range(5, 6, 1)))
    miniBatch.slice(1, 1).getTarget() should be (Tensor[Float](1).fill(1))
    miniBatch.slice(2, 1).getTarget() should be (Tensor[Float](1).fill(2))
    miniBatch.slice(3, 1).getTarget() should be (Tensor[Float](1).fill(3))
  }

}
