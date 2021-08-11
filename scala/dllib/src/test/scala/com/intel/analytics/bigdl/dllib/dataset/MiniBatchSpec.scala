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

  "SparseTensorMiniBatch set" should "return right result" in {
    val a1 = Tensor.sparse(Tensor[Float](4).range(1, 4, 1))
    val a2 = Tensor.sparse(Tensor[Float](4).range(5, 8, 1))
    val b1 = Tensor[Float](5).range(1, 5, 1)
    val b2 = Tensor[Float](5).range(6, 10, 1)
    val c1 = Tensor[Float](1).fill(1)
    val c2 = Tensor[Float](1).fill(0)
    val sample1 = TensorSample[Float](Array(a1, b1), Array(c1))
    val sample2 = TensorSample[Float](Array(a2, b2), Array(c2))
    val miniBatch = SparseMiniBatch[Float](2, 1)
    miniBatch.set(Array(sample1, sample2))

    val input = miniBatch.getInput()
    val target = miniBatch.getTarget()

    val expectedInput1 = Tensor.sparse(Array(Array(0, 0, 0, 0, 1, 1, 1, 1),
      Array(0, 1, 2, 3, 0, 1, 2, 3)),
      Array.range(1, 9).map(_.toFloat), Array(2, 4))
    val expectedInput2 = Tensor[Float].range(1, 10).reshape(Array(2, 5))
    input.toTable[Tensor[Float]](1) should be (expectedInput1)
    input.toTable[Tensor[Float]](2) should be (expectedInput2)

    val expectedTarget = Tensor[Float](T(1.0f, 0.0f)).reshape(Array(2, 1))
    target should be (expectedTarget)
  }

  "SparseTensorMiniBatch with different TensorTypes" should "return right result" in {
    val a1 = Tensor.sparse(Tensor[Double](4).range(1, 4, 1))
    val a2 = Tensor.sparse(Tensor[Double](4).range(5, 8, 1))
    val b1 = Tensor[String](5)
      .setValue(1, "a").setValue(2, "b")
      .setValue(3, "c").setValue(4, "d").setValue(5, "e")
    val b2 = Tensor[String](5)
      .setValue(1, "1").setValue(2, "2")
      .setValue(3, "3").setValue(4, "4").setValue(5, "5")
    val c1 = Tensor[Double](1).fill(1)
    val c2 = Tensor[Double](1).fill(0)
    val sample1 = TensorSample.create[Float](Array(a1, b1), Array(c1))
    val sample2 = TensorSample.create[Float](Array(a2, b2), Array(c2))
    val miniBatch = SparseMiniBatch[Float](2, 1)
    miniBatch.set(Array(sample1, sample2))

    val input = miniBatch.getInput()
    val target = miniBatch.getTarget()

    val expectedInput1 = Tensor.sparse(Array(Array(0, 0, 0, 0, 1, 1, 1, 1),
      Array(0, 1, 2, 3, 0, 1, 2, 3)),
      Array.range(1, 9).map(_.toFloat), Array(2, 4))
    input.toTable[Tensor[Double]](1) should be (expectedInput1)
    input.toTable[Tensor[String]](2).storage().array() should be (Array(
      "a", "b", "c", "d", "e", "1", "2", "3", "4", "5"))

    val expectedTarget = Tensor[Double](T(1.0, 0.0)).reshape(Array(2, 1))
    target should be (expectedTarget)
  }

}
