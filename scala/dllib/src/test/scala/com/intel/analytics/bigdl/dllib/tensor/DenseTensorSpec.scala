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
import spire.syntax.module

@com.intel.analytics.bigdl.tags.Parallel
class DenseTensorSpec extends FlatSpec with Matchers {

  "Construct with empty parameter" should "be empty" in {
    val t: Tensor[Double] = new DenseTensor[Double]()
    t.nDimension should be(0)
  }

  "Test type of Activity" should "be correct" in {
    val t: Tensor[Double] = new DenseTensor[Double]()
    t.isTensor should be(true)
    t.isTable should be(false)
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
    val t: Tensor[Double] = new DenseTensor(new ArrayStorage(storage))
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
      i = i + 1;
      i
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
    val t: Tensor[Double] = new DenseTensor(new ArrayStorage(Array(3.0, 4, 5)))
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
      i = i + 1;
      i
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
    val src: Tensor[Double] = new DenseTensor(new ArrayStorage(Array(8.0, 9)))
    t(2) = src
    t(Array(1, 1)) should be(1)
    t(Array(1, 2)) should be(1)
    t(Array(2, 1)) should be(8)
    t(Array(2, 2)) should be(9)
    t(Array(3, 1)) should be(1)
    t(Array(3, 2)) should be(1)
  }

  "One index update a 1d-dimension tensor" should "update the value" in {
    val t: Tensor[Double] = new DenseTensor(new ArrayStorage(Array(3.0, 4, 5)))
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
      i = i + 1;
      i
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

    val src: Tensor[Double] = new DenseTensor(new ArrayStorage(Array(9.0, 10)))

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
      i = i + 1;
      i
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
    val t: Tensor[Double] = new DenseTensor(new ArrayStorage(Array(1.0, 2, 3)))
    val t1 = t.clone()
    t.isSameSizeAs(t1) should be(true)
    t1.isContiguous() should be(true)
    t.valueAt(1) should be(1)
    t.valueAt(2) should be(2)
    t.valueAt(3) should be(3)
  }

  "resize as" should "get the correct tensor" in {
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

  "resize 0 tensor" should "work properly" in {
    val t = Tensor[Double](10)
    t.resize(0)
    t.nElement() should be (0)
    t.dim() should be (1)
  }

  "resize 2d tensor to 0" should "work properly" in {
    val t = Tensor[Double](2, 10)
    t.resize(0, 10)
    t.nElement() should be (0)
    t.dim() should be (2)

    val t1 = Tensor[Double](2, 10)
    t1.resize(10, 0)
    t1.nElement() should be (0)
    t1.dim() should be (2)
  }

  "resize 0 element 2d tensor to > 0 element" should "work properly" in {
    val t = Tensor[Double](0, 10)
    t.resize(1, 10)
    t.nElement() should be (10)
    t.dim() should be (2)

    val t1 = Tensor[Double](10, 0)
    t1.resize(10, 1)
    t1.nElement() should be (10)
    t1.dim() should be (2)
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
      i = i + 1;
      i
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
      i = i + 1;
      i
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
      i = i + 1;
      i
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
      i = i + 1;
      i
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
      i = i + 1;
      i
    })
    val t1: Tensor[Double] = new DenseTensor[Double](2, 2)
    i = 0
    t1.apply1(v => {
      i = i + 1;
      i
    })

    t.map(t1, (a, b) => a * b)
    t(Array(1, 1)) should be(1)
    t(Array(1, 2)) should be(4)
    t(Array(2, 1)) should be(9)
    t(Array(2, 2)) should be(16)
  }

  "equals" should "be correct" in {
    val t: Tensor[Double] = new DenseTensor(new ArrayStorage(Array(1.0, 2, 3)))
    val t1: Tensor[Double] = new DenseTensor(new ArrayStorage(Array(1.0, 2, 3)))
    val t2: Tensor[Double] = new DenseTensor(new ArrayStorage(Array(1.0, 2, 4)))

    t == t1 should be(true)
    t == t2 should be(false)
  }

  "hashCode" should "be correct" in {
    val t: Tensor[Double] = new DenseTensor(new ArrayStorage(Array(1.0, 2, 3)))
    val t1: Tensor[Double] = new DenseTensor(new ArrayStorage(Array(1.0, 2, 3)))
    val t2: Tensor[Double] = new DenseTensor(new ArrayStorage(Array(1.0, 2, 4)))

    t.hashCode() == t1.hashCode() should be(true)
    t.hashCode() == t2.hashCode() should be(false)
  }

  "toString" should "be correct" in {
    var t: Tensor[Double] = new DenseTensor[Double]()
    val EMPTY_STRING =
      """Empty Tensor"""
    t.toString should be(EMPTY_STRING)

    t = Tensor.scalar[Double](1)
    t.toString should be("Scalar(1.0)")

    t = new DenseTensor(new ArrayStorage(Array(1.0, 2.0, 3.0)))
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
      i = i + 1
      i
    })
    t.toString should be(MATRIX_STRING)

    t = new DenseTensor(2, 5, 3, 4)
    i = 0
    t.apply1(v => {
      i = i + 1
      i
    })
    println(t)
  }

  "toString" should "be elegant if the tensor is too large" in {
    val t = new DenseTensor[Float](1000)
    var i = 0
    t.apply1(v => {
      i = i + 1; i
    })
    val OneD_STRING =
      "1.0\n2.0\n3.0\n...\n998.0\n999.0\n1000.0\n" +
        "[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 1000]"
    t.toString should be(OneD_STRING)
    val s = new DenseTensor[Float](50, 50)
    i = 0
    s.apply1(v => {
      i = i + 1; i
    })
    val MATRIX_STRING =
      "1.0\t2.0\t3.0\t...\t48.0\t49.0\t50.0\t\n" +
        "51.0\t52.0\t53.0\t...\t98.0\t99.0\t100.0\t\n" +
        "101.0\t102.0\t103.0\t...\t148.0\t149.0\t150.0\t\n" +
        "...\n" +
        "2351.0\t2352.0\t2353.0\t...\t2398.0\t2399.0\t2400.0\t\n" +
        "2401.0\t2402.0\t2403.0\t...\t2448.0\t2449.0\t2450.0\t\n" +
        "2451.0\t2452.0\t2453.0\t...\t2498.0\t2499.0\t2500.0\t\n" +
        "[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 50x50]"
    s.toString should be(MATRIX_STRING)
    val r = new DenseTensor[Float](1, 10, 50, 50)
    i = 0
    r.apply1(v => {
      i = i + 1; i
    })
    val MULTIPLE_MATRIX_STRING =
      "(1,1,.,.) =\n" +
      "1.0\t2.0\t3.0\t...\t48.0\t49.0\t50.0\t\n" +
        "51.0\t52.0\t53.0\t...\t98.0\t99.0\t100.0\t\n" +
        "101.0\t102.0\t103.0\t...\t148.0\t149.0\t150.0\t\n" +
        "...\n" +
        "2351.0\t2352.0\t2353.0\t...\t2398.0\t2399.0\t2400.0\t\n" +
        "2401.0\t2402.0\t2403.0\t...\t2448.0\t2449.0\t2450.0\t\n" +
        "2451.0\t2452.0\t2453.0\t...\t2498.0\t2499.0\t2500.0\t\n\n" +
      "(1,2,.,.) =\n" +
        "2501.0\t2502.0\t2503.0\t...\t2548.0\t2549.0\t2550.0\t\n" +
        "2551.0\t2552.0\t2553.0\t...\t2598.0\t2599.0\t2600.0\t\n" +
        "2601.0\t2602.0\t2603.0\t...\t2648.0\t2649.0\t2650.0\t\n" +
        "...\n" +
        "4851.0\t4852.0\t4853.0\t...\t4898.0\t4899.0\t4900.0\t\n" +
        "4901.0\t4902.0\t4903.0\t...\t4948.0\t4949.0\t4950.0\t\n" +
        "4951.0\t4952.0\t4953.0\t...\t4998.0\t4999.0\t5000.0\t\n\n" +
      "...\n\n" +
      "(1,9,.,.) =\n" +
        "20001.0\t20002.0\t20003.0\t...\t20048.0\t20049.0\t20050.0\t\n" +
        "20051.0\t20052.0\t20053.0\t...\t20098.0\t20099.0\t20100.0\t\n" +
        "20101.0\t20102.0\t20103.0\t...\t20148.0\t20149.0\t20150.0\t\n" +
        "...\n" +
        "22351.0\t22352.0\t22353.0\t...\t22398.0\t22399.0\t22400.0\t\n" +
        "22401.0\t22402.0\t22403.0\t...\t22448.0\t22449.0\t22450.0\t\n" +
        "22451.0\t22452.0\t22453.0\t...\t22498.0\t22499.0\t22500.0\t\n\n" +
      "(1,10,.,.) =\n" +
        "22501.0\t22502.0\t22503.0\t...\t22548.0\t22549.0\t22550.0\t\n" +
        "22551.0\t22552.0\t22553.0\t...\t22598.0\t22599.0\t22600.0\t\n" +
        "22601.0\t22602.0\t22603.0\t...\t22648.0\t22649.0\t22650.0\t\n" +
        "...\n" +
        "24851.0\t24852.0\t24853.0\t...\t24898.0\t24899.0\t24900.0\t\n" +
        "24901.0\t24902.0\t24903.0\t...\t24948.0\t24949.0\t24950.0\t\n" +
        "24951.0\t24952.0\t24953.0\t...\t24998.0\t24999.0\t25000.0\t\n\n" +
    "[com.intel.analytics.bigdl.tensor.DenseTensor$mcF$sp of size 1x10x50x50]"
    r.toString should be(MULTIPLE_MATRIX_STRING)
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
      i += 1;
      i
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
    val tensor = new DenseTensor[Double](
      new ArrayStorage[Double](Array(1.0, 2, 3, 4)), 1, Array(2, 2))
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
    val tensor = new DenseTensor[Double](new ArrayStorage(Array(1.0, 2, 3, 4)))
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
    val tensor = new DenseTensor(new ArrayStorage(Array(1.0, 2, 3, 4)), 1, Array(2, 2))
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
    val tensor = new DenseTensor(new ArrayStorage(Array(1.0, 2, 3, 4)))
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

  "squeezeNewTensor" should "work correclty" in {
    val input = Tensor[Float](4, 10).fill(1.0f)
    val output = input.narrow(1, 2, 1).squeezeNewTensor()

    output.size() should be(Array(10))
    input should be(Tensor[Float](4, 10).fill(1.0f))
  }

  "tensor apply on 1D tensor" should "work correctly" in {
    val a = Tensor[Float](4)
    a.rand()
    val b = a(2)
    b.storageOffset() should be(2)
    b.size() should be(Array())
    b.stride() should be(Array())
    b.nElement() should be(1)

    b.setValue(0.01f)
    b.value() should be(0.01f)
    a.valueAt(2) should be(0.01f)
  }

  "tensor apply on 2D tensor" should "work correctly" in {
    val a = Tensor[Float](3, 4)
    a.rand()
    val b = a(2)
    b.storageOffset() should be(5)
    b.size() should be(Array(4))
    b.stride() should be(Array(1))
    b.nElement() should be(4)

    b.setValue(3, 0.01f)
    b.valueAt(3) should be(0.01f)
    a.valueAt(2, 3) should be(0.01f)
  }

  "Scalar tensor" should "be able to construct" in {
    val t: Tensor[Double] = DenseTensor[Double](1.0)
    t.nDimension should be(0)
    t.size().isEmpty should be(true)
  }

  "Scalar tensor" should "not have size" in {
    val t: Tensor[Double] = DenseTensor[Double](1.0)
    val thrown = intercept[Exception] {
      t.size(1)
    }
    thrown.isInstanceOf[IllegalArgumentException] should be(true)
  }

  "Scalar tensor" should "be able to add" in {
    val t: Tensor[Double] = DenseTensor[Double](1.0)
    val y: Tensor[Double] = DenseTensor[Double](1.0)
    t.add(1.0, y)
    t should be(DenseTensor[Double](2.0))
  }

  "Scalar tensor" should "be able to set value" in {
    val t: Tensor[Double] = DenseTensor[Double](1.0)
    t.setValue(2.0)
    t should be(DenseTensor[Double](2.0))
  }

  "Scalar tensor" should "be able to calc max" in {
    val t: Tensor[Double] = DenseTensor[Double](1.0)
    t.max() should be(1.0)
  }

  "Scalar tensor" should "be able to calc min" in {
    val t: Tensor[Double] = DenseTensor[Double](1.0)
    t.max() should be(1.0)
  }

  "Scalar tensor" should "be able to calc nElement" in {
    val t: Tensor[Double] = DenseTensor[Double](1.0)
    t.nElement() should be(1)
  }

  "Scalar tensor" should "be able to get element" in {
    val t: Tensor[Double] = DenseTensor[Double](1.0)
    t.apply(Array[Int]()) should be(1.0)
  }

  "Scalar tensor" should "be able to update" in {
    val t: Tensor[Double] = DenseTensor[Double](1.0)
    t.update(Array[Int](), 2.0)
    t should be(DenseTensor[Double](2.0))
  }

  "Tensor add and sub" should "support broadcasting" in {
    val t1 = Tensor[Double](T(1, 2, 3))
    val t2 = Tensor[Double](T(T(2, 5, 3), T(3, 6, 4)))
    t2.add(t1) should be (Tensor[Double](T(T(3, 7, 6), T(4, 8, 7))))
  }

  "Tensor sub" should "support broadcasting" in {
    val t1 = Tensor[Double](T(1, 2, 3))
    val t2 = Tensor[Double](T(T(2, 5, 3), T(3, 6, 4)))
    t2.sub(t1) should be (Tensor[Double](T(T(1, 3, 0), T(2, 4, 1))))
  }

  "Tensor div" should "support broadcasting" in {
    val t1 = Tensor[Double](T(1, 2, 3))
    val t2 = Tensor[Double](T(T(2, 4, 6), T(3, 6, 9)))
    t2.div(t1) should be (Tensor[Double](T(T(2, 2, 2), T(3, 3, 3))))
  }

  "Tensor add " should "support broadcasting 2" in {
    val t1 = Tensor[Double](T(
      T(
        T(1, 2, 3),
        T(4, 5, 6)
      ),
      T(
        T(2, 1, 6),
        T(5, 4, 3)
      ),
      T(
        T(4, 1, 3),
        T(4, 5, 3)
      )
    ))
    val t2 = Tensor[Double](T(
      T(
        T(2),
        T(3)
      )
    ))

    val cloneT1 = t1.clone()
    val oldStorage = t1.storage()
    t1.add(t2) should be(Tensor[Double](T(
      T(
        T(3, 4, 5),
        T(7, 8, 9)
      ),
      T(
        T(4, 3, 8),
        T(8, 7, 6)
      ),
      T(
        T(6, 3, 5),
        T(7, 8, 6)
      )
    )))
    oldStorage.eq(t1.storage()) should be(true)

    t2.add(cloneT1) should be(Tensor[Double](T(
      T(
        T(3, 4, 5),
        T(7, 8, 9)
      ),
      T(
        T(4, 3, 8),
        T(8, 7, 6)
      ),
      T(
        T(6, 3, 5),
        T(7, 8, 6)
      )
    )))
  }

  "Tensor add" should "support broadcasting with singleton dimension" in {
    val t1 = Tensor[Double](T(T(1, 2, 3)))
    val t2 = Tensor[Double](T(T(2, 5, 3), T(3, 6, 4)))
    t2.add(t1) should be(Tensor[Double](T(T(3, 7, 6), T(4, 8, 7))))
  }

  "Tensor add" should "catch exception when broadcasting size not match" in {
    val t1 = Tensor[Double](T(1, 2))
    val t2 = Tensor[Double](T(T(2, 5, 3), T(3, 6, 4)))
    intercept[IllegalArgumentException] {
      t2.add(t1) should be(Tensor[Double](T(T(3, 7, 6), T(4, 8, 7))))
    }
  }

  "Tensor add" should "catch exception when broadcasting size not match 2" in {
    val t1 = Tensor[Double](T(T(1, 2, 3), T(1, 2, 3), T(1, 2, 3)))
    val t2 = Tensor[Double](T(T(2, 5, 3), T(3, 6, 4)))
    intercept[IllegalArgumentException] {
      t2.add(t1) should be(Tensor[Double](T(T(3, 7, 6), T(4, 8, 7))))
    }
  }

  "Select on a Vector " should "be a scalar" in {
    val t: Tensor[Double] = new DenseTensor[Double](2)
    val result = t.select(1, 1)
    result.isScalar should be(true)
  }

  "Negative" should "be correct" in {
    val x = Tensor[Float](T(1, 2, 3))
    val y = Tensor[Float](3)
    y.negative(x) should be(Tensor[Float](T(-1, -2, -3)))
    x.negative(x) should be(Tensor[Float](T(-1, -2, -3)))
  }

  "Floor" should "be correct" in {
    val x = Tensor[Float](T(1.1, 2.5, 3.7))
    x.floor() should be(Tensor[Float](T(1, 2, 3)))

    val y = Tensor[Double](T(1.1, 2.5, 3.7))
    y.floor() should be(Tensor[Double](T(1, 2, 3)))

    val z = Tensor[Int](T(1, 2, 3))
    z.floor() should be(Tensor[Int](T(1, 2, 3)))
  }

  "mm" should "be correct when multiply twice" in {
    val o = Tensor[Float]().resize(2, 2)
    val x = Tensor[Float](T(T(1, 2, 3), T(4, 5, 6)))
    val y = Tensor[Float](T(T(1, 2), T(3, 4), T(5, 6)))
    o.mm(x, y)
    o.mm(x, y)
    o should be(Tensor[Float](T(T(22, 28), T(49, 64))))
  }

  "set" should "work properly" in {
    val t = Tensor[Float](1, 3)
    t.set()
    t.size() should be (Array[Int]())
    t.nElement() should be (0)
    t.dim() should be (0)
  }

  "cast" should "work on scalar" in {
    val scalar1 = Tensor.scalar[Float](1.0f)
    val scalar2 = Tensor.scalar[Int](0)
    scalar1.cast[Int](scalar2)

    scalar2.value() should be(1)
  }

  "size" should "work on empty tensor" in {
    val t = Tensor[Float]()
    t.size() should be (null)
  }

  "stride" should "work on empty tensor" in {
    val t = Tensor[Float]()
    t.stride() should be (null)
  }
}
