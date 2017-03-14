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

import breeze.linalg.{DenseMatrix => BrzDenseMatrix, DenseVector => BrzDenseVector}
import com.intel.analytics.bigdl.nn.Linear
import com.intel.analytics.bigdl.utils.T
import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector}
import org.scalatest.{FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Parallel
class DenseTensorSpec extends FlatSpec with Matchers {

  "Construct with empty parameter" should "be empty" in {
    val t: Tensor[Double] = new DenseTensor[Double]()
    t.nDimension should be(0)
  }

  "Construct with dimension list" should "return correct value" in {
    val t: Tensor[Double] = new DenseTensor[Double](1, 2, 3)
    t.nDimension should be(3)
    t.size().length should be(3)
    t.size(1) should be(1)
    t.size(2) should be(2)
    t.size(3) should be(3)
    t.stride().length should be(3)
    t.stride(1) should be(6)
    t.stride(2) should be(3)
    t.stride(3) should be(1)
  }

  "Construct with storage" should "return 1D vector" in {
    val storage = Array(1.0, 2.0, 3.0)
    val t: Tensor[Double] = new DenseTensor(Storage(storage))
    t.nDimension should be(1)
    t.size().length should be(1)
    t.size(1) should be(3)
    t.stride().length should be(1)
    t.stride(1) should be(1)
  }

  "Construct A Tensor with Table directly" should "work correclty " in {
    val t: Tensor[Double] = Tensor[Double](
      T(
        T(
          T(1.0, 2.0),
          T(3.0, 4.0)
        ),
        T(
          T(5.0, 6.0),
          T(7.0, 8.0)
        )
      ))

    t(Array(1, 1, 1)) should be(1)
    t(Array(1, 1, 2)) should be(2)
    t(Array(1, 2, 1)) should be(3)
    t(Array(1, 2, 2)) should be(4)
    t(Array(2, 1, 1)) should be(5)
    t(Array(2, 1, 2)) should be(6)
    t(Array(2, 2, 1)) should be(7)
    t(Array(2, 2, 2)) should be(8)
  }

  "Construct with another tensor" should "use same storage and separate views" in {
    val other = new DenseTensor[Double](1, 2, 3)
    val t: Tensor[Double] = new DenseTensor(other)
    t.nDimension() should be(3)
    t.size(1) should be(1)
    t.size(2) should be(2)
    t.size(3) should be(3)
    t.stride().length should be(3)
    t.stride(1) should be(6)
    t.stride(2) should be(3)
    t.stride(3) should be(1)
  }

  "Fill with one" should "make every element be one" in {
    val t: Tensor[Double] = new DenseTensor[Double](1, 2, 5)
    t.fill(1)
    val s = t.storage().asInstanceOf[Storage[Double]]
    for (e <- s) {
      e should be(1)
    }
  }

  "Transpose" should "be correct" in {
    val t: Tensor[Double] = new DenseTensor[Double](3, 3)
    var i = 0
    t.apply1(v => {
      i = i + 1; i
    })
    t(Array(1, 1)) should be(1)
    t(Array(1, 2)) should be(2)
    t(Array(1, 3)) should be(3)
    t(Array(2, 1)) should be(4)
    t(Array(2, 2)) should be(5)
    t(Array(2, 3)) should be(6)
    t(Array(3, 1)) should be(7)
    t(Array(3, 2)) should be(8)
    t(Array(3, 3)) should be(9)

    val t1 = t.t
    t1(Array(1, 1)) should be(1)
    t1(Array(1, 2)) should be(4)
    t1(Array(1, 3)) should be(7)
    t1(Array(2, 1)) should be(2)
    t1(Array(2, 2)) should be(5)
    t1(Array(2, 3)) should be(8)
    t1(Array(3, 1)) should be(3)
    t1(Array(3, 2)) should be(6)
    t1(Array(3, 3)) should be(9)
  }

  "One index on a multi-dimension tensor" should "return subset" in {
    val t: Tensor[Double] = new DenseTensor[Double](3, 2).fill(1)
    t(2).fill(0)
    t(Array(1, 1)) should be(1)
    t(Array(1, 2)) should be(1)
    t(Array(2, 1)) should be(0)
    t(Array(2, 2)) should be(0)
    t(Array(3, 1)) should be(1)
    t(Array(3, 2)) should be(1)
  }

  "One index on a 1d-dimension tensor" should "return value" in {
    val t: Tensor[Double] = new DenseTensor(Storage(Array(3.0, 4, 5)))
    t.valueAt(2) should be(4.0)
  }

  "Query for value" should "get tensor" in {
    val t: Tensor[Double] = new DenseTensor[Double](3, 2, 3).fill(1)
    val result = t(Array(1, 2, 3))
    result should be(1)
  }

  "Query with table" should "get correct value" in {
    val t: Tensor[Double] = new DenseTensor[Double](3, 2, 2)
    var i = 0
    t.apply1(v => {
      i = i + 1; i
    })

    t(T(2, 2, 1)) should be(new DenseTensor[Double](1).fill(7))

    val subset: Tensor[Double] = t(T(2, T(), 2))
    subset.valueAt(1) should be(6)
    subset.valueAt(2) should be(8)
  }

  "One index update a multi-dimension tensor" should "fill a subset" in {
    val t: Tensor[Double] = new DenseTensor[Double](3, 2).fill(1)
    t(2) = 0
    t(Array(1, 1)) should be(1)
    t(Array(1, 2)) should be(1)
    t(Array(2, 1)) should be(0)
    t(Array(2, 2)) should be(0)
    t(Array(3, 1)) should be(1)
    t(Array(3, 2)) should be(1)
  }

  "One index update a multi-dimension tensor with tensor" should
    "copy the tensor to the subset" in {
    val t: Tensor[Double] = new DenseTensor[Double](3, 2).fill(1)
    val src: Tensor[Double] = new DenseTensor(Storage(Array(8.0, 9)))
    t(2) = src
    t(Array(1, 1)) should be(1)
    t(Array(1, 2)) should be(1)
    t(Array(2, 1)) should be(8)
    t(Array(2, 2)) should be(9)
    t(Array(3, 1)) should be(1)
    t(Array(3, 2)) should be(1)
  }

  "One index update a 1d-dimension tensor" should "update the value" in {
    val t: Tensor[Double] = new DenseTensor(Storage(Array(3.0, 4, 5)))
    t(2) = 6
    t.valueAt(1) should be(3.0)
    t.valueAt(2) should be(6.0)
    t.valueAt(3) should be(5.0)
  }

  "Array index update a tensor" should "update the value" in {
    val t: Tensor[Double] = new DenseTensor[Double](3, 2).fill(1)
    t(Array(2, 2)) = 2
    t(Array(1, 1)) should be(1)
    t(Array(1, 2)) should be(1)
    t(Array(2, 1)) should be(1)
    t(Array(2, 2)) should be(2)
    t(Array(3, 1)) should be(1)
    t(Array(3, 2)) should be(1)
  }

  "Update with table" should "get correct value" in {
    val t: Tensor[Double] = new DenseTensor[Double](3, 2)
    var i = 0
    t.apply1(v => {
      i = i + 1; i
    })
    t(T(2, 2)) = 0
    t(Array(1, 1)) should be(1)
    t(Array(1, 2)) should be(2)
    t(Array(2, 1)) should be(3)
    t(Array(2, 2)) should be(0)
    t(Array(3, 1)) should be(5)
    t(Array(3, 2)) should be(6)


    t(T(T(2, 3), 1)) = 7
    t(Array(1, 1)) should be(1)
    t(Array(1, 2)) should be(2)
    t(Array(2, 1)) should be(7)
    t(Array(2, 2)) should be(0)
    t(Array(3, 1)) should be(7)
    t(Array(3, 2)) should be(6)

    val src: Tensor[Double] = new DenseTensor(Storage(Array(9.0, 10)))

    t(T(T(2, 3), 1)) = src
    t(Array(1, 1)) should be(1)
    t(Array(1, 2)) should be(2)
    t(Array(2, 1)) should be(9)
    t(Array(2, 2)) should be(0)
    t(Array(3, 1)) should be(10)
    t(Array(3, 2)) should be(6)
  }

  "Criteria update a tensor" should "update the value" in {
    val t: Tensor[Double] = new DenseTensor[Double](3, 2)
    var i = 0
    t.apply1(v => {
      i = i + 1; i
    })
    val criteria: Double => Boolean = v => v >= 4
    t(criteria) = 0
    t(Array(1, 1)) should be(1)
    t(Array(1, 2)) should be(2)
    t(Array(2, 1)) should be(3)
    t(Array(2, 2)) should be(0)
    t(Array(3, 1)) should be(0)
    t(Array(3, 2)) should be(0)
  }

  "Contiguous" should "get correct answer" in {
    val t: Tensor[Double] = new DenseTensor[Double](2, 3).fill(1)
    t.isContiguous() should be(true)
    val t1 = t.contiguous()
    val r = t.t()
    r.isContiguous() should be(false)
    val r1 = r.contiguous()
    r1.isContiguous() should be(true)
    t(Array(2, 3)) = 2
    t1(Array(2, 3)) should be(2)
    r(Array(3, 2)) should be(2)
    r1(Array(3, 2)) should be(1)
  }

  "isSameSizeAs" should "return correct value" in {
    val t: Tensor[Double] = new DenseTensor[Double](2, 4)
    val t1: Tensor[Double] = new DenseTensor[Double](2, 4)
    val t2 = t1.t()

    t.isSameSizeAs(t1) should be(true)
    t.isSameSizeAs(t2) should be(false)
  }

  "clone" should "get a seperated tensor" in {
    val t: Tensor[Double] = new DenseTensor(Storage(Array(1.0, 2, 3)))
    val t1 = t.clone()
    t.isSameSizeAs(t1) should be(true)
    t1.isContiguous() should be(true)
    t.valueAt(1) should be(1)
    t.valueAt(2) should be(2)
    t.valueAt(3) should be(3)
  }

  "resise as" should "get the correct tensor" in {
    val t: Tensor[Double] = new DenseTensor[Double](3, 4)
    val t1: Tensor[Double] = new DenseTensor[Double](5, 5)
    val t2: Tensor[Double] = new DenseTensor[Double](2, 2)

    t1.resizeAs(t)
    t1.isSameSizeAs(t) should be(true)
    t2.resizeAs(t)
    t2.isSameSizeAs(t) should be(true)
  }

  "resize as" should "get the correct tensor when resizing to a larger tensor" in {
    val t1: Tensor[Double] = new DenseTensor[Double](3, 4)
    val t: Tensor[Double] = new DenseTensor[Double](1, 1)
    t.resizeAs(t1)
    t.storage().size should be(t1.storage().size)
  }

  "nElement" should "return correct value" in {
    val t: Tensor[Double] = new DenseTensor[Double](3, 4)
    t.nElement() should be(12)
    t(T(T(), T(2, 3))).nElement() should be(6)
  }

  "select" should "be correct" in {
    val t: Tensor[Double] = new DenseTensor[Double](3, 2)
    var i = 0
    t.apply1(v => {
      i = i + 1; i
    })
    val t1 = t.select(1, 2)
    t1.nDimension() should be(1)
    t1.size(1) should be(2)
    t1(Array(1)) should be(3)
    t1(Array(2)) should be(4)
  }

  "storage" should "be correct" in {
    val t: Tensor[Double] = new DenseTensor[Double](3, 2)
    var i = 0
    t.apply1(v => {
      i = i + 1; i
    })
    val s = t.storage().asInstanceOf[Storage[Double]]
    var j = 0
    for (e <- s) {
      j = j + 1
      e should be(j)
    }
  }

  "storageOffset" should "be correct" in {
    val t: Tensor[Double] = new DenseTensor[Double](3, 2)
    t.storageOffset() should be(1)
    val t1 = t.select(1, 2)
    t1.storageOffset() should be(3)
  }

  "narrow" should "return correct answer" in {
    val t: Tensor[Double] = new DenseTensor[Double](3, 3)
    var i = 0
    t.apply1(v => {
      i = i + 1; i
    })
    val t1 = t.narrow(1, 2, 2)
    t1.nDimension() should be(2)
    t1.size(1) should be(2)
    t1.size(2) should be(3)
    t1(Array(1, 1)) should be(4)
    t1(Array(1, 2)) should be(5)
    t1(Array(1, 3)) should be(6)
    t1(Array(2, 1)) should be(7)
    t1(Array(2, 2)) should be(8)
    t1(Array(2, 3)) should be(9)
  }

  "copy" should "return correct answer" in {
    val t: Tensor[Double] = new DenseTensor[Double](4).fill(1)
    val t1: Tensor[Double] = new DenseTensor[Double](2, 2).copy(t)
    t1(Array(1, 1)) should be(1)
    t1(Array(1, 2)) should be(1)
    t1(Array(2, 1)) should be(1)
    t1(Array(2, 2)) should be(1)
  }

  it should "return correct answer for continuous tensor" in {
    val t: Tensor[Double] = new DenseTensor[Double](2, 3).rand()
    val t1: Tensor[Double] = new DenseTensor[Double](2, 3).fill(0)
    t1.copy(t)
    t should be(t1)
  }

  it should "return correct answer for non-continuous tensor" in {
    val t: Tensor[Double] = new DenseTensor[Double](2, 3).rand().t
    val t1: Tensor[Double] = new DenseTensor[Double](3, 2).fill(0)
    t1.copy(t)
    t should be(t1)
  }

  "apply1" should "be correct" in {
    val t: Tensor[Double] = new DenseTensor[Double](2, 2)
    var i = 0
    t.apply1(v => {
      i = i + 1; i
    })
    t(Array(1, 1)) should be(1)
    t(Array(1, 2)) should be(2)
    t(Array(2, 1)) should be(3)
    t(Array(2, 2)) should be(4)
  }

  "map" should "be correct" in {
    val t: Tensor[Double] = new DenseTensor[Double](2, 2)
    var i = 0
    t.apply1(v => {
      i = i + 1; i
    })
    val t1: Tensor[Double] = new DenseTensor[Double](2, 2)
    i = 0
    t1.apply1(v => {
      i = i + 1; i
    })

    t.map(t1, (a, b) => a * b)
    t(Array(1, 1)) should be(1)
    t(Array(1, 2)) should be(4)
    t(Array(2, 1)) should be(9)
    t(Array(2, 2)) should be(16)
  }

  "equals" should "be correct" in {
    val t: Tensor[Double] = new DenseTensor(Storage(Array(1.0, 2, 3)))
    val t1: Tensor[Double] = new DenseTensor(Storage(Array(1.0, 2, 3)))
    val t2: Tensor[Double] = new DenseTensor(Storage(Array(1.0, 2, 4)))

    t == t1 should be(true)
    t == t2 should be(false)
  }

  "hashCode" should "be correct" in {
    val t: Tensor[Double] = new DenseTensor(Storage(Array(1.0, 2, 3)))
    val t1: Tensor[Double] = new DenseTensor(Storage(Array(1.0, 2, 3)))
    val t2: Tensor[Double] = new DenseTensor(Storage(Array(1.0, 2, 4)))

    t.hashCode() == t1.hashCode() should be(true)
    t.hashCode() == t2.hashCode() should be(false)
  }

  "toString" should "be correct" in {
    var t: Tensor[Double] = new DenseTensor[Double]()
    val EMPTY_STRING =
      """[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp with no dimension]"""
    t.toString should be(EMPTY_STRING)
    t = new DenseTensor(Storage(Array(1.0, 2.0, 3.0)))
    val OneD_STRING =
      "1.0\n" +
        "2.0\n" +
        "3.0\n" +
        "[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 3]"
    t.toString should be(OneD_STRING)
    val MATRIX_STRING =
      "1.0\t2.0\t3.0\t4.0\t\n" +
        "5.0\t6.0\t7.0\t8.0\t\n" +
        "9.0\t10.0\t11.0\t12.0\t\n" +
        "[com.intel.analytics.bigdl.tensor.DenseTensor$mcD$sp of size 3x4]"
    t = new DenseTensor[Double](3, 4)
    var i = 0
    t.apply1(v => {
      i = i + 1; i
    })
    t.toString should be(MATRIX_STRING)

    t = new DenseTensor(2, 5, 3, 4)
    i = 0
    t.apply1(v => {
      i = i + 1; i
    })
    println(t)
  }

  "squeeze" should "be correct" in {
    var t: Tensor[Double] = new DenseTensor[Double](3, 1, 2, 1)
    t.squeeze()
    t.nDimension() should be(2)
    t.size(1) should be(3)
    t.size(2) should be(2)


    t = new DenseTensor(3, 1, 2, 1)
    t.squeeze(3)
    t.nDimension() should be(4)
    t.size(1) should be(3)
    t.size(2) should be(1)
    t.size(3) should be(2)
    t.size(4) should be(1)

    t.squeeze(2)
    t.nDimension() should be(3)
    t.size(1) should be(3)
    t.size(2) should be(2)
    t.size(3) should be(1)

  }

  "unfold" should "be correct" in {
    val x = Tensor[Double](3)
    for (i <- 1 to 3) x(i) = i

    x.unfold(1, 2, 1)(Array(1, 1)) should be(1)
    x.unfold(1, 2, 1)(Array(1, 2)) should be(2)
    x.unfold(1, 2, 1)(Array(2, 1)) should be(2)
    x.unfold(1, 2, 1)(Array(2, 2)) should be(3)
    x.unfold(1, 2, 1).size(1) should be(2)
    x.unfold(1, 2, 1).size(2) should be(2)
    x.unfold(1, 2, 1).stride(1) should be(1)
    x.unfold(1, 2, 1).stride(2) should be(1)
    x.unfold(1, 2, 1).nDimension() should be(2)

    x.unfold(1, 2, 2)(Array(1, 1)) should be(1)
    x.unfold(1, 2, 2)(Array(1, 2)) should be(2)
    x.unfold(1, 2, 2).size(1) should be(1)
    x.unfold(1, 2, 2).size(2) should be(2)
    x.unfold(1, 2, 2).stride(1) should be(2)
    x.unfold(1, 2, 2).stride(2) should be(1)
    x.unfold(1, 2, 2).nDimension() should be(2)
  }

  "randperm" should "be correct" in {
    val a = Tensor.randperm[Double](1)
    a.size(1) should be(1)
    a(Array(1)) should be(1.0)
  }

  "expand" should "be correct" in {
    val x = Tensor[Double](3, 1)
    var i = 0
    x.apply1(e => {
      i += 1; i
    })

    val result = x.expand(Array(3, 2))

    result.nDimension() should be(2)
    result.size(1) should be(3)
    result.size(2) should be(2)

    result(Array(1, 1)) should be(1)
    result(Array(1, 2)) should be(1)
    result(Array(2, 1)) should be(2)
    result(Array(2, 2)) should be(2)
    result(Array(3, 1)) should be(3)
    result(Array(3, 2)) should be(3)
  }

  "repreatTensor" should "generate right tensor" in {
    val x = Tensor[Double](3)
    for (i <- 1 to 3) x(i) = i
    val result = Tensor.repeatTensor[Double](x, 2, 2)

    result.nDimension() should be(2)
    result.size(1) should be(2)
    result.size(2) should be(6)

    result(Array(1, 1)) should be(1)
    result(Array(1, 2)) should be(2)
    result(Array(1, 3)) should be(3)
    result(Array(1, 4)) should be(1)
    result(Array(1, 5)) should be(2)
    result(Array(1, 6)) should be(3)
    result(Array(2, 1)) should be(1)
    result(Array(2, 2)) should be(2)
    result(Array(2, 3)) should be(3)
    result(Array(2, 4)) should be(1)
    result(Array(2, 5)) should be(2)
    result(Array(2, 6)) should be(3)

    val result2 = Tensor.repeatTensor(x, 3, 2)
  }

  "Tensor to BreezeMatrix" should "correct" in {
    val tensor = new DenseTensor[Double](Storage[Double](Array(1.0, 2, 3, 4)), 1, Array(2, 2))
    val matrix = tensor.toBreezeMatrix()
    matrix.isTranspose should be(true)
    matrix(0, 0) should be(1.0)
    matrix(0, 1) should be(2.0)
    matrix(1, 0) should be(3.0)
    matrix(1, 1) should be(4.0)

    // Tensor is row-major but breeze is col-major. So if tensor is transposed,
    // breeze is not, vice-versa
    val matrixTransposed = tensor.t.toBreezeMatrix()
    matrixTransposed.isTranspose should be(false)
    matrixTransposed(0, 0) should be(1.0)
    matrixTransposed(0, 1) should be(3.0)
    matrixTransposed(1, 0) should be(2.0)
    matrixTransposed(1, 1) should be(4.0)
  }

  "BreezeMatrix to Tensor" should "correct" in {
    val matrix = BrzDenseMatrix((1.0, 2.0), (3.0, 4.0))
    val tensor = Tensor(matrix)
    tensor(Array(1, 1)) should be(1.0)
    tensor(Array(1, 2)) should be(2.0)
    tensor(Array(2, 1)) should be(3.0)
    tensor(Array(2, 2)) should be(4.0)

    val matrixTranspose = BrzDenseMatrix((1.0, 2.0), (3.0, 4.0)).t
    val tensorTranspose = Tensor(matrixTranspose)
    tensorTranspose(Array(1, 1)) should be(1.0)
    tensorTranspose(Array(1, 2)) should be(3.0)
    tensorTranspose(Array(2, 1)) should be(2.0)
    tensorTranspose(Array(2, 2)) should be(4.0)
  }

  "Tensor to BreezeVector" should "correct" in {
    val tensor = new DenseTensor[Double](Storage(Array(1.0, 2, 3, 4)))
    val vector = tensor.toBreezeVector()
    vector(0) should be(1.0)
    vector(1) should be(2.0)
    vector(2) should be(3.0)
    vector(3) should be(4.0)
  }

  "BreezeVector to Tensor" should "correct" in {
    val vector = new BrzDenseVector(Array(1.0, 2.0, 3.0, 4.0))
    val tensor = Tensor(vector)
    tensor(Array(1)) should be(1.0)
    tensor(Array(2)) should be(2.0)
    tensor(Array(3)) should be(3.0)
    tensor(Array(4)) should be(4.0)
  }

  "Tensor to MLMatrix" should "correct" in {
    val tensor = new DenseTensor(Storage(Array(1.0, 2, 3, 4)), 1, Array(2, 2))
    val matrix = tensor.toMLlibMatrix()
    matrix.isTransposed should be(true)
    matrix(0, 0) should be(1.0)
    matrix(0, 1) should be(2.0)
    matrix(1, 0) should be(3.0)
    matrix(1, 1) should be(4.0)

    // Tensor is row-major but breeze is col-major. So if tensor is transposed, breeze is not,
    // vice-versa
    val matrixTransposed = tensor.t.toMLlibMatrix()
    matrixTransposed.isTransposed should be(false)
    matrixTransposed(0, 0) should be(1.0)
    matrixTransposed(0, 1) should be(3.0)
    matrixTransposed(1, 0) should be(2.0)
    matrixTransposed(1, 1) should be(4.0)
  }

  "MLMatrix to Tensor" should "correct" in {
    val matrix = new DenseMatrix(2, 2, Array(1.0, 2.0, 3.0, 4.0))
    val tensor = Tensor(matrix)
    tensor(Array(1, 1)) should be(1.0)
    tensor(Array(2, 1)) should be(2.0)
    tensor(Array(1, 2)) should be(3.0)
    tensor(Array(2, 2)) should be(4.0)

    val matrixTranspose = new DenseMatrix(2, 2, Array(1.0, 2.0, 3.0, 4.0), true)
    val tensorTranspose = Tensor(matrixTranspose)
    tensorTranspose(Array(1, 1)) should be(1.0)
    tensorTranspose(Array(1, 2)) should be(3.0)
    tensorTranspose(Array(2, 1)) should be(2.0)
    tensorTranspose(Array(2, 2)) should be(4.0)
  }

  "Tensor to MLVector" should "correct" in {
    val tensor = new DenseTensor(Storage(Array(1.0, 2, 3, 4)))
    val vector = tensor.toMLlibVector()
    vector(0) should be(1.0)
    vector(1) should be(2.0)
    vector(2) should be(3.0)
    vector(3) should be(4.0)
  }

  "MLVector to Tensor" should "correct" in {
    val vector = new DenseVector(Array(1.0, 2.0, 3.0, 4.0))
    val tensor = Tensor(vector)
    tensor(Array(1)) should be(1.0)
    tensor(Array(2)) should be(2.0)
    tensor(Array(3)) should be(3.0)
    tensor(Array(4)) should be(4.0)
  }

  "Reshape" should "works correctly" in {
    val tensor = Tensor.randperm[Double](20)
    tensor.resize(2, 10)
    val shape = tensor.reshape(Array(2, 2, 5)).size()
    shape(0) should be(2)
    shape(1) should be(2)
    shape(2) should be(5)
  }

  "addSingletonDimension" should "works correclty" in {
    val tensor = Tensor[Float](3, 4)
    tensor.addSingletonDimension()
    tensor.nDimension() should be(3)
    tensor.size(1) should be(1)
    tensor.size(2) should be(3)
    tensor.size(3) should be(4)

    tensor.addSingletonDimension(tensor, 3)
    tensor.nDimension() should be(4)
    tensor.size(1) should be(1)
    tensor.size(2) should be(3)
    tensor.size(3) should be(1)
    tensor.size(4) should be(4)
  }

  "toTensor" should "work correclty" in {
    val input = Tensor[Float](3, 4)
    val module = Linear[Float](4, 5)
    val output = module.forward(input)

    output.toTensor[Float] should be(output)
    try {
      output.toTensor[Double]
      fail()
    } catch {
      case ex: IllegalArgumentException =>
    }
  }

  "toTable" should "work correclty" in {
    val input = Tensor[Float](3, 4)
    val module = Linear[Float](4, 5)
    val output = module.forward(input)

    try {
      output.toTable
      fail()
    } catch {
      case ex: IllegalArgumentException =>
    }
  }
}
