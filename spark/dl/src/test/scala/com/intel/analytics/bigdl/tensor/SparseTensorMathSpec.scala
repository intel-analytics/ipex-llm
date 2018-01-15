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
class SparseTensorMathSpec extends FlatSpec with Matchers {
  "Sparse Matrix * Dense Vector" should "be correct" in {
    val sparseM = Tensor.sparse(Tensor(2, 3).range(1, 12, 2))
    val a = Tensor(3)
    a.setValue(1, 2)
    a.setValue(3, 1)
    val res = Tensor(2)
    SparseTensorMath.addmv[Float](res, 1, res, 1, sparseM, a)
    val correctRes = Tensor(2)
    correctRes.setValue(1, 7)
    correctRes.setValue(2, 25)

    res shouldEqual correctRes
  }

  "Sparse Matrix * Dense Vector" should "be correct 2" in {
    val sparseM = Tensor.sparse(Tensor(2, 3).range(1, 12, 2))
    val a = Tensor(3)
    a.setValue(1, 2)
    a.setValue(3, 1)
    val res = Tensor(2).fill(1)
    SparseTensorMath.addmv[Float](res, 2, res, 3, sparseM, a)
    val correctRes = Tensor(2)
    correctRes.setValue(1, 23)
    correctRes.setValue(2, 77)

    res shouldEqual correctRes
  }

  "narrowed Sparse Matrix * Dense Vector" should "be correct" in {
    val sparseM = Tensor.sparse(Tensor(4, 3).range(1, 12, 1).narrow(1, 2, 2))
    val a = Tensor(3)
    a.setValue(1, 2)
    a.setValue(3, 1)
    val res = Tensor(2).fill(1)
    SparseTensorMath.addmv[Float](res, 1, res, 1, sparseM, a)
    val correctRes = Tensor(2)
    correctRes.setValue(1, 15)
    correctRes.setValue(2, 24)

    res shouldEqual correctRes
  }

  "Sparse Matrix * transposed Dense Matrix" should "be correct" in {
    val sparseM = Tensor.sparse(Tensor(2, 3).setValue(1, 3, 1).setValue(2, 2, 1))
    val denseM = Tensor(2, 3).range(1, 12, 2).t()

    val res = Tensor(2, 2).fill(10)
    SparseTensorMath.addmm[Float](res, 1, res, 1, sparseM, denseM)
    val correctRes = Tensor(2, 2)
    correctRes.setValue(1, 1, 15)
    correctRes.setValue(1, 2, 21)
    correctRes.setValue(2, 1, 13)
    correctRes.setValue(2, 2, 19)

    res shouldEqual correctRes
  }

  "Sparse Matrix * transposed Dense Matrix" should "be correct 2" in {
    val sparseM = Tensor.sparse(Tensor(2, 3).setValue(1, 3, 1).setValue(2, 2, 1))
    val denseM = Tensor(2, 3).range(1, 12, 2).t()

    val res = Tensor(2, 2).fill(10)
    SparseTensorMath.addmm[Float](res, 0, res, 1, sparseM, denseM)
    val correctRes = Tensor(2, 2)
    correctRes.setValue(1, 1, 5)
    correctRes.setValue(1, 2, 11)
    correctRes.setValue(2, 1, 3)
    correctRes.setValue(2, 2, 9)

    res shouldEqual correctRes
  }

  "Sparse Matrix * transposed Dense Matrix" should "be correct 3" in {
    val sparseM = Tensor.sparse(Tensor(2, 3).setValue(1, 3, 1).setValue(2, 2, 1))
    val denseM = Tensor(2, 3).range(1, 12, 2).t()

    val res = Tensor(2, 2).fill(10)
    SparseTensorMath.addmm[Float](res, 0, res, 2, sparseM, denseM)
    val correctRes = Tensor(2, 2)
    correctRes.setValue(1, 1, 10)
    correctRes.setValue(1, 2, 22)
    correctRes.setValue(2, 1, 6)
    correctRes.setValue(2, 2, 18)

    res shouldEqual correctRes
  }

  "Sparse Matrix * Dense Matrix" should "be correct" in {
    val sparseM = Tensor.sparse(Tensor(2, 3).setValue(1, 3, 1).setValue(2, 2, 1))
    val denseM = Tensor(3, 2).range(1, 12, 2)

    val res = Tensor(2, 2).fill(10)
    SparseTensorMath.addmm[Float](res, 1, res, 1, sparseM, denseM)
    val correctRes = Tensor(2, 2)
    correctRes.setValue(1, 1, 19)
    correctRes.setValue(1, 2, 21)
    correctRes.setValue(2, 1, 15)
    correctRes.setValue(2, 2, 17)

    res shouldEqual correctRes
  }

  "Sparse Matrix * Dense Matrix" should "be correct 2" in {
    val sparseM = Tensor.sparse(Tensor(2, 3).setValue(1, 3, 1).setValue(2, 2, 1))
    val denseM = Tensor(3, 2).range(1, 12, 2)

    val res = Tensor(2, 2).fill(10)
    SparseTensorMath.addmm[Float](res, 0, res, 1, sparseM, denseM)
    val correctRes = Tensor(2, 2)
    correctRes.setValue(1, 1, 9)
    correctRes.setValue(1, 2, 11)
    correctRes.setValue(2, 1, 5)
    correctRes.setValue(2, 2, 7)

    res shouldEqual correctRes
  }

  "Sparse Matrix * Dense Matrix" should "be correct 3" in {
    val sparseM = Tensor.sparse(Tensor(2, 3).setValue(1, 3, 1).setValue(2, 2, 1))
    val denseM = Tensor(3, 2).range(1, 12, 2)

    val res = Tensor(2, 2).fill(10)
    SparseTensorMath.addmm[Float](res, 0, res, 2, sparseM, denseM)
    val correctRes = Tensor(2, 2)
    correctRes.setValue(1, 1, 18)
    correctRes.setValue(1, 2, 22)
    correctRes.setValue(2, 1, 10)
    correctRes.setValue(2, 2, 14)

    res shouldEqual correctRes
  }

  "Dense Matrix * Sparse Matrix" should "be correct" in {
    val sparseM = Tensor.sparse(Tensor(3, 2).setValue(2, 2, 1).setValue(3, 1, 1))
    val denseM = Tensor(2, 3).range(1, 12, 2)

    val res = Tensor(2, 2).fill(10)
    SparseTensorMath.addmm[Float](res, 1, res, 1, denseM, sparseM)
    val correctRes = Tensor(2, 2)
    correctRes.setValue(1, 1, 15)
    correctRes.setValue(1, 2, 13)
    correctRes.setValue(2, 1, 21)
    correctRes.setValue(2, 2, 19)

    res shouldEqual correctRes
  }

  "Dense Matrix * Sparse Matrix" should "be correct 2" in {
    val sparseM = Tensor.sparse(Tensor(3, 2).setValue(2, 2, 1).setValue(3, 1, 1))
    val denseM = Tensor(2, 3).range(1, 12, 2)

    val res = Tensor(2, 2).fill(10)
    SparseTensorMath.addmm[Float](res, 0, res, 1, denseM, sparseM)
    val correctRes = Tensor(2, 2)
    correctRes.setValue(1, 1, 5)
    correctRes.setValue(1, 2, 3)
    correctRes.setValue(2, 1, 11)
    correctRes.setValue(2, 2, 9)

    res shouldEqual correctRes
  }

  "Dense Matrix * Sparse Matrix" should "be correct 3" in {
    val sparseM = Tensor.sparse(Tensor(3, 2).setValue(2, 2, 1).setValue(3, 1, 1))
    val denseM = Tensor(2, 3).range(1, 12, 2)

    val res = Tensor(2, 2).fill(10)
    SparseTensorMath.addmm[Float](res, 0, res, 2, denseM, sparseM)
    val correctRes = Tensor(2, 2)
    correctRes.setValue(1, 1, 10)
    correctRes.setValue(1, 2, 6)
    correctRes.setValue(2, 1, 22)
    correctRes.setValue(2, 2, 18)

    res shouldEqual correctRes
  }

  "Transposed Dense Matrix * Sparse Matrix" should "be correct" in {
    val sparseM = Tensor.sparse(Tensor(3, 2).setValue(2, 2, 1).setValue(3, 1, 1))
    val denseM = Tensor(3, 2).range(1, 12, 2).t()

    val res = Tensor(2, 2).fill(10)
    SparseTensorMath.addmm[Float](res, 1, res, 1, denseM, sparseM)
    val correctRes = Tensor(2, 2)
    correctRes.setValue(1, 1, 19)
    correctRes.setValue(1, 2, 15)
    correctRes.setValue(2, 1, 21)
    correctRes.setValue(2, 2, 17)

    res shouldEqual correctRes
  }

  "Transposed Dense Matrix * Sparse Matrix" should "be correct 2" in {
    val sparseM = Tensor.sparse(Tensor(3, 2).setValue(2, 2, 1).setValue(3, 1, 1))
    val denseM = Tensor(3, 2).range(1, 12, 2).t()

    val res = Tensor(2, 2).fill(10)
    SparseTensorMath.addmm[Float](res, 0, res, 1, denseM, sparseM)
    val correctRes = Tensor(2, 2)
    correctRes.setValue(1, 1, 9)
    correctRes.setValue(1, 2, 5)
    correctRes.setValue(2, 1, 11)
    correctRes.setValue(2, 2, 7)

    res shouldEqual correctRes
  }

  "Transposed Dense Matrix * Sparse Matrix" should "be correct 3" in {
    val sparseM = Tensor.sparse(Tensor(3, 2).setValue(2, 2, 1).setValue(3, 1, 1))
    val denseM = Tensor(3, 2).range(1, 12, 2).t()

    val res = Tensor(2, 2).fill(10)
    SparseTensorMath.addmm[Float](res, 0, res, 2, denseM, sparseM)
    val correctRes = Tensor(2, 2)
    correctRes.setValue(1, 1, 18)
    correctRes.setValue(1, 2, 10)
    correctRes.setValue(2, 1, 22)
    correctRes.setValue(2, 2, 14)

    res shouldEqual correctRes
  }

  "Dense Matrix * narrowed Sparse Matrix" should "be correct" in {
    val sparseM = Tensor.sparse(Tensor(5, 2)
        .setValue(1, 2, 3).setValue(2, 1, 4).setValue(3, 2, 1)
      .setValue(4, 1, 1).setValue(5, 1, 3)).narrow(1, 2, 3)
    val denseM = Tensor(4, 3).range(1, 12, 1)

    val res = Tensor(4, 2).fill(10)
    SparseTensorMath.addmm[Float](res, 0, res, 2, denseM, sparseM)
    val correctRes = Tensor(4, 2)
    correctRes.setValue(1, 1, 14)
    correctRes.setValue(1, 2, 4)
    correctRes.setValue(2, 1, 44)
    correctRes.setValue(2, 2, 10)
    correctRes.setValue(3, 1, 74)
    correctRes.setValue(3, 2, 16)
    correctRes.setValue(4, 1, 104)
    correctRes.setValue(4, 2, 22)

    res shouldEqual correctRes
  }

  "narrowed Sparse Matrix * Dense Matrix" should "be correct" in {
    val sparseM = Tensor.sparse(Tensor(5, 2)
      .setValue(1, 2, 3).setValue(2, 1, 4).setValue(3, 2, 1)
      .setValue(4, 1, 1).setValue(5, 1, 3)).narrow(1, 2, 3)
    val denseM = Tensor(2, 4).range(1, 8, 1)

    val res = Tensor(3, 4).fill(10)
    SparseTensorMath.addmm[Float](res, 1, res, 1, sparseM, denseM)
    val correctRes = Tensor(3, 4)
    correctRes.setValue(1, 1, 14)
    correctRes.setValue(1, 2, 18)
    correctRes.setValue(1, 3, 22)
    correctRes.setValue(1, 4, 26)
    correctRes.setValue(2, 1, 15)
    correctRes.setValue(2, 2, 16)
    correctRes.setValue(2, 3, 17)
    correctRes.setValue(2, 4, 18)
    correctRes.setValue(3, 1, 11)
    correctRes.setValue(3, 2, 12)
    correctRes.setValue(3, 3, 13)
    correctRes.setValue(3, 4, 14)

    res shouldEqual correctRes
  }
}
