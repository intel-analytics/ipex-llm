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

import com.intel.analytics.bigdl.utils.T
import org.scalatest.{FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Parallel
class DenseTensorMathSpec extends FlatSpec with Matchers {
  "a.dist(b, 1)" should "be correct" in {
    val a: Tensor[Double] = new DenseTensor(new ArrayStorage(Array(1.0, 2.0, 3.0)))
    val b: Tensor[Double] = new DenseTensor(new ArrayStorage(Array(2.0, 3.0, 4.0)))

    a.dist(b, 1) should equal(3)
  }

  "a.dist(b, 2)" should "be correct" in {
    val a: Tensor[Double] = new DenseTensor(new ArrayStorage(Array(1.0, 2.0, 3.0)))
    val b: Tensor[Double] = new DenseTensor(new ArrayStorage(Array(3.0, 4.0, 5.0)))

    a.dist(b, 2) should equal(math.sqrt(12))
  }

  "a.dist(b, 3)" should "be correct" in {
    val a: Tensor[Double] = new DenseTensor(new ArrayStorage(Array(1.0, 2.0, 3.0)))
    val b: Tensor[Double] = new DenseTensor(new ArrayStorage(Array(3.0, 4.0, 5.0)))

    a.dist(b, 3) should equal(math.pow(24, 1.0 / 3))
  }

  "vector + scalar" should "be correct" in {
    val s = 2.0
    val v: Tensor[Double] = new DenseTensor(new ArrayStorage(Array(1.0, 2.0, 3.0)))
    val r = v + s
    r(Array(1)) should be(3.0)
    r(Array(2)) should be(4.0)
    r(Array(3)) should be(5.0)
  }

  "vector + vector" should "be correct" in {
    val v1: Tensor[Double] = new DenseTensor(new ArrayStorage(Array(1.0, 2.0, 3.0)))
    val v2: Tensor[Double] = new DenseTensor(new ArrayStorage(Array(1.0, 2.0, 3.0)))
    val r = v1 + v2
    r(Array(1)) should be(2.0)
    r(Array(2)) should be(4.0)
    r(Array(3)) should be(6.0)
  }

  "vector + vector which is not contiguous" should "be correct" in {
    val v1: Tensor[Double] = new DenseTensor[Double](2, 4).fill(1)
    v1.t()
    val v2: Tensor[Double] = new DenseTensor(new ArrayStorage(
      Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)))
    val r = v1 + v2
    r(Array(1, 1)) should be(2.0)
    r(Array(1, 2)) should be(3.0)
    r(Array(1, 3)) should be(4.0)
    r(Array(1, 4)) should be(5.0)
    r(Array(2, 1)) should be(6.0)
    r(Array(2, 2)) should be(7.0)
    r(Array(2, 3)) should be(8.0)
    r(Array(2, 4)) should be(9.0)
  }

  "vector - scalar" should "be correct" in {
    val s = 2.0
    val v: Tensor[Double] = new DenseTensor(new ArrayStorage(Array(1.0, 2.0, 3.0)))
    val r = v - s
    r(Array(1)) should be(-1.0)
    r(Array(2)) should be(0.0)
    r(Array(3)) should be(1.0)
  }

  "vector - vector" should "be correct" in {
    val v1: Tensor[Double] = new DenseTensor(new ArrayStorage(Array(1.0, 2.0, 3.0)))
    val v2: Tensor[Double] = new DenseTensor(new ArrayStorage(Array(2.0, 0.0, -1.0)))
    val r = v1 - v2
    r(Array(1)) should be(-1.0)
    r(Array(2)) should be(2.0)
    r(Array(3)) should be(4.0)
  }

  "vector * scalar" should "be correct" in {
    val s = 2.0
    val v: Tensor[Double] = new DenseTensor(new ArrayStorage(Array(1.0, 2.0, 3.0)))
    val r = v * s
    r(Array(1)) should be(2.0)
    r(Array(2)) should be(4.0)
    r(Array(3)) should be(6.0)
  }

  "vector * vector" should "be correct" in {
    val v1: Tensor[Double] = new DenseTensor(new ArrayStorage(Array(1.0, 2.0, 3.0)))
    val v2: Tensor[Double] = new DenseTensor(new ArrayStorage(Array(2.0, 0.0, -1.0)))
    val r = v1 * v2
    r(Array(1)) should be(-1.0)
  }

  "matrix * vector" should "be correct" in {
    val mat: Tensor[Double] = new DenseTensor(2, 3)
    mat(Array(1, 1)) = 2
    mat(Array(1, 2)) = 4
    mat(Array(1, 3)) = 3
    mat(Array(2, 1)) = 5
    mat(Array(2, 2)) = 6
    mat(Array(2, 3)) = 1

    val vec: Tensor[Double] = new DenseTensor(new ArrayStorage(Array(3.0, 1, 1)))
    val r = mat * vec
    r(Array(1)) should be(13.0)
    r(Array(2)) should be(22.0)
  }

  "transpose matrix * vector" should "be correct" in {
    val mat: Tensor[Double] = new DenseTensor(3, 2)
    mat(Array(1, 1)) = 2
    mat(Array(1, 2)) = 4
    mat(Array(2, 1)) = 3
    mat(Array(2, 2)) = 5
    mat(Array(3, 1)) = 6
    mat(Array(3, 2)) = 1

    val mat1 = mat.t

    val vec: Tensor[Double] = new DenseTensor(new ArrayStorage(Array(3.0, 1, 1)))
    val r = mat1 * vec
    r(Array(1)) should be(15.0)
    r(Array(2)) should be(18.0)
  }

  "uncontiguous matrix * vector" should "be correct" in {
    val tensor: Tensor[Double] = new DenseTensor(3, 2, 2)
    tensor(Array(1, 1, 1)) = 2
    tensor(Array(1, 2, 1)) = 4
    tensor(Array(2, 1, 1)) = 3
    tensor(Array(2, 2, 1)) = 5
    tensor(Array(3, 1, 1)) = 6
    tensor(Array(3, 2, 1)) = 1

    val matrix = tensor(T(T(), T(), 1)).t()

    val vec: Tensor[Double] = new DenseTensor(new ArrayStorage(Array(3.0, 1, 1)))
    val r = matrix * vec
    r(Array(1)) should be(15.0)
    r(Array(2)) should be(18.0)
  }

  "matrix * matrix" should "be correct" in {
    val mat1: Tensor[Double] = new DenseTensor(3, 2)
    var i = 0
    mat1.apply1(_ => {
      i = i + 1;
      i
    })
    val mat2: Tensor[Double] = new DenseTensor(2, 3)
    i = 0
    mat2.apply1(_ => {
      i = i + 1;
      i
    })
    val r = mat2 * mat1
    r(Array(1, 1)) should be(22)
    r(Array(1, 2)) should be(28)
    r(Array(2, 1)) should be(49)
    r(Array(2, 2)) should be(64)
  }

  "transpose matrix * matrix" should "be correct" in {
    val mat1: Tensor[Double] = new DenseTensor(3, 2)
    var i = 0
    mat1.apply1(_ => {
      i = i + 1;
      i
    })
    val mat2: Tensor[Double] = new DenseTensor(3, 2)
    i = 0
    mat2.apply1(_ => {
      i = i + 1;
      i
    })
    val r = mat2.t * mat1
    r(Array(1, 1)) should be(35)
    r(Array(1, 2)) should be(44)
    r(Array(2, 1)) should be(44)
    r(Array(2, 2)) should be(56)
  }

  "matrix * transpose matrix" should "be correct" in {
    val mat1: Tensor[Double] = new DenseTensor(2, 3)
    var i = 0
    mat1.apply1(_ => {
      i = i + 1;
      i
    })
    val mat2: Tensor[Double] = new DenseTensor(2, 3)
    i = 0
    mat2.apply1(_ => {
      i = i + 1;
      i
    })
    val r = mat2 * mat1.t
    r(Array(1, 1)) should be(14)
    r(Array(1, 2)) should be(32)
    r(Array(2, 1)) should be(32)
    r(Array(2, 2)) should be(77)
  }

  "transpose matrix * transpose matrix" should "be correct" in {
    val mat1: Tensor[Double] = new DenseTensor(3, 2)
    var i = 0
    mat1.apply1(_ => {
      i = i + 1;
      i
    })
    val mat2: Tensor[Double] = new DenseTensor(2, 3)
    i = 0
    mat2.apply1(_ => {
      i = i + 1;
      i
    })
    val r = mat1.t * mat2.t
    r(Array(1, 1)) should be(22)
    r(Array(1, 2)) should be(49)
    r(Array(2, 1)) should be(28)
    r(Array(2, 2)) should be(64)
  }

  "noncontiguous matrix * noncontiguous matrix" should "be correct" in {
    val tensor: Tensor[Double] = new DenseTensor(3, 2, 2)
    tensor(Array(1, 1, 1)) = 1
    tensor(Array(1, 2, 1)) = 2
    tensor(Array(2, 1, 1)) = 3
    tensor(Array(2, 2, 1)) = 4
    tensor(Array(3, 1, 1)) = 5
    tensor(Array(3, 2, 1)) = 6

    val mat1: Tensor[Double] = tensor(T(T(), T(), 1)).t
    val mat2: Tensor[Double] = tensor(T(T(), T(), 1))

    val r = mat1 * mat2
    r(Array(1, 1)) should be(35)
    r(Array(1, 2)) should be(44)
    r(Array(2, 1)) should be(44)
    r(Array(2, 2)) should be(56)
  }

  "vector / scalar" should "be correct" in {
    val s = 2.0
    val v: Tensor[Double] = new DenseTensor(new ArrayStorage(Array(1.0, 2.0, 3.0)))
    val r = v / s
    r(Array(1)) should be(0.5)
    r(Array(2)) should be(1.0)
    r(Array(3)) should be(1.5)
  }

  "vector / vector" should "be correct" in {
    val v1: Tensor[Double] = new DenseTensor(new ArrayStorage(Array(1.0, 2.0, 3.0)))
    val v2: Tensor[Double] = new DenseTensor(new ArrayStorage(Array(2.0, 1.0, -1.0)))
    val r = v1 / v2
    r(Array(1)) should be(0.5)
    r(Array(2)) should be(2.0)
    r(Array(3)) should be(-3.0)
  }

  "-vector" should "be correct" in {
    val v: Tensor[Double] = new DenseTensor(new ArrayStorage(Array(1.0, 2.0, 3.0)))
    val r = -v
    r(Array(1)) should be(-1.0)
    r(Array(2)) should be(-2.0)
    r(Array(3)) should be(-3.0)
  }

  "max operation" should "return correct value" in {
    val t: Tensor[Double] = new DenseTensor(3, 3)
    var i = 0
    t.apply1(v => {
      i = i + 1;
      i
    })

    t.max() should be(9)
  }

  "max with dim" should "return correct value" in {
    val test = Tensor[Double](Storage(Array(1.0, 2, 3, 4, 5, 6, 7, 8)), 1, Array(2, 4))
    val (values1, indices1) = test.max(1)
    values1 should be(Tensor[Double](Storage(Array(5.0, 6, 7, 8)), 1, Array(1, 4)))
    indices1 should be(Tensor[Double](Storage(Array(2.0, 2, 2, 2)), 1, Array(1, 4)))

    val (values2, indices2) = test.max(2)
    values2 should be(Tensor[Double](Storage(Array(4.0, 8.0)), 1, Array(2, 1)))
    indices2 should be(Tensor[Double](Storage(Array(4.0, 4)), 1, Array(2, 1)))
  }

  "max with dim on 1d tensor" should "return correct value" in {
    val test = Tensor[Double](Storage(Array(1.0, 2, 3, 4, 5, 6, 7, 8)))
    val (values, indices) = test.max(1)
    values should be(Tensor[Double](Storage(Array(8.0))))
    indices should be(Tensor[Double](Storage(Array(8.0))))
  }

  "sum operation" should "return correct value" in {
    val t: Tensor[Double] = new DenseTensor(2, 3)
    var i = 0
    t.apply1(e => {
      i = i + 1;
      i
    })
    t.sum() should be(21)

    val result1 = t.sum(1)
    result1.size(1) should be(1)
    result1.size(2) should be(3)

    result1(Array(1, 1)) should be(5)
    result1(Array(1, 2)) should be(7)
    result1(Array(1, 3)) should be(9)


    val result2 = t.sum(2)
    result2.size(1) should be(2)
    result2.size(2) should be(1)

    result2(Array(1, 1)) should be(6)
    result2(Array(2, 1)) should be(15)
  }

  "addmm" should "return correct value" in {
    val a_data = Array(
      1.0, 2, 3, 4,
      1, 2, 3, 4,
      1, 2, 3, 4
    )
    val a = new DenseTensor[Double](new ArrayStorage(a_data), 1, Array(3, 4))


    val b_data = Array(
      1.0, 2,
      1, 2,
      1, 2,
      1, 2
    )
    val b = new DenseTensor[Double](new ArrayStorage(b_data), 1, Array(4, 2))

    val c = Tensor[Double]()
    c.resize(Array(3, 2))
    c.addmm(a, b)

    val expect_c_data = Array(
      10.0, 20.0,
      10, 20,
      10, 20
    )

    val expect_c = new DenseTensor[Double](new ArrayStorage(expect_c_data), 1, Array(3, 2))
    c.map(expect_c, (a, b) => {
      a should be(b +- 1e-6)
      a
    })
  }

  "addmm plus another tensor" should "return correct value" in {
    val a_data = Array(
      1.0, 2, 3, 4,
      1, 2, 3, 4,
      1, 2, 3, 4
    )
    val a = new DenseTensor[Double](new ArrayStorage(a_data), 1, Array(3, 4))


    val b_data = Array(
      1.0, 2,
      1, 2,
      1, 2,
      1, 2
    )
    val b = new DenseTensor[Double](new ArrayStorage(b_data), 1, Array(4, 2))

    val m_data = Array(
      1.0, 2,
      1, 2,
      1, 2
    )
    val m = new DenseTensor[Double](new ArrayStorage(m_data), 1, Array(3, 2))

    val c = Tensor[Double]()
    c.addmm(m, a, b)

    val expect_c_data = Array(
      11.0, 22.0,
      11, 22,
      11, 22
    )

    val expect_c = new DenseTensor[Double](new ArrayStorage(expect_c_data), 1, Array(3, 2))
    c.map(expect_c, (a, b) => {
      a should be(b +- 1e-6)
      a
    })
  }

  "addr transpose" should "return correct value" in {
    val v1: Tensor[Double] = new DenseTensor(new ArrayStorage(Array(1.0, 2.0, 3.0)))
    val v2: Tensor[Double] = new DenseTensor(new ArrayStorage(Array(2.0, 0.0, -1.0)))
    val tensor: Tensor[Double] = new DenseTensor(3, 3)
    tensor(Array(1, 1)) = 1
    tensor(Array(1, 2)) = 2
    tensor(Array(1, 3)) = 2
    tensor(Array(2, 1)) = 3
    tensor(Array(2, 2)) = 4
    tensor(Array(2, 3)) = 4
    tensor(Array(3, 1)) = 5
    tensor(Array(3, 2)) = 6
    tensor(Array(3, 3)) = 6
    val mat: Tensor[Double] = tensor.t
    val r = Tensor[Double]()
    r.resize(Array(3, 3))
    r.addr(1.0, mat, 1.0, v1, v2)
    val expect_r = new DenseTensor(new ArrayStorage(Array(3.0, 3.0, 4.0,
      6.0, 4.0, 4.0,
      8.0, 4.0, 3.0)), 1, Array(3, 3))
    r should be (expect_r)
  }

  "addr" should "return correct value" in {
    val v1: Tensor[Double] = new DenseTensor(new ArrayStorage(Array(1.0, 2.0, 3.0)))
    val v2: Tensor[Double] = new DenseTensor(new ArrayStorage(Array(2.0, 0.0, -1.0)))
    val r = Tensor[Double]()
    r.resize(Array(3, 3))
    r.addr(v1, v2)
    r should be (new DenseTensor[Double](new ArrayStorage(Array(2.0, 0.0, -1.0,
      4.0, 0.0, -2.0,
      6.0, 0.0, -3.0)), 1, Array(3, 3)))
  }

  "addr noncontiguous" should "return correct value" in {
    val v1: Tensor[Double] = new DenseTensor(new ArrayStorage(Array(1.0, 2.0, 3.0)))
    val v2: Tensor[Double] = new DenseTensor(new ArrayStorage(Array(2.0, 0.0, -1.0)))
    val tensor: Tensor[Double] = new DenseTensor(3, 3, 2)
    tensor(Array(1, 1, 1)) = 1
    tensor(Array(1, 2, 1)) = 2
    tensor(Array(1, 3, 1)) = 2
    tensor(Array(2, 1, 1)) = 3
    tensor(Array(2, 2, 1)) = 4
    tensor(Array(2, 3, 1)) = 4
    tensor(Array(3, 1, 1)) = 5
    tensor(Array(3, 2, 1)) = 6
    tensor(Array(3, 3, 1)) = 6
    val mat: Tensor[Double] = tensor(T(T(), T(), 1)).t

    val r = Tensor[Double]()
    r.resize(Array(3, 3))
    r.addr(1, mat, 1, v1, v2)
    r should be (new DenseTensor[Double](new ArrayStorage(Array(3.0, 3.0, 4.0,
      6.0, 4.0, 4.0,
      8.0, 4.0, 3.0)), 1, Array(3, 3)))
  }

  "uniform" should "return correct value" in {
    val t = Tensor[Double]()
    for (i <- 0 to 1000) {
      val rand = t.uniform()
      rand should be(0.5 +- 0.5)
    }
  }

  "uniform(n)" should "return correct value" in {
    val t = Tensor[Double]()
    t.uniform(1.0) should be(1.0)
    for (i <- 0 to 1000) {
      val rand = t.uniform(11.0)
      rand should be(6.0 +- 5.0)
    }
  }

  "uniform(l, n)" should "return correct value" in {
    val t = Tensor[Double]()
    t.uniform(1.0, 1.0) should be(1.0)
    t.uniform(-2.0, -2.0) should be(-2.0)
    for (i <- 0 to 1000) {
      val rand = t.uniform(-11.0, 11.0)
      rand should be(0.0 +- 11.0)
    }
  }

  "mean operation" should "return correct value" in {
    val t: Tensor[Double] = new DenseTensor(2, 3)
    var i = 0
    t.apply1(e => {
      i = i + 1;
      i
    })
    t.mean() should be(3.5)

    val result1 = t.mean(1)
    result1.size(1) should be(1)
    result1.size(2) should be(3)

    result1(Array(1, 1)) should be(2.5)
    result1(Array(1, 2)) should be(3.5)
    result1(Array(1, 3)) should be(4.5)


    val result2 = t.mean(2)
    result2.size(1) should be(2)
    result2.size(2) should be(1)

    result2(Array(1, 1)) should be(2)
    result2(Array(2, 1)) should be(5)
  }

  "mean operation on 3D tensor" should "return correct value" in {
    val t: Tensor[Double] = new DenseTensor(2, 3, 4)
    var i = 0
    t.apply1(e => {
      i = i + 1;
      i
    })
    t.mean() should be(12.5)

    val result1 = t.mean(1)
    result1.size(1) should be(1)
    result1.size(2) should be(3)
    result1.size(3) should be(4)

    result1(Array(1, 1, 1)) should be(7)
    result1(Array(1, 1, 2)) should be(8)
    result1(Array(1, 1, 3)) should be(9)
    result1(Array(1, 1, 4)) should be(10)
    result1(Array(1, 2, 1)) should be(11)
    result1(Array(1, 2, 2)) should be(12)
    result1(Array(1, 2, 3)) should be(13)
    result1(Array(1, 2, 4)) should be(14)
    result1(Array(1, 3, 1)) should be(15)
    result1(Array(1, 3, 2)) should be(16)
    result1(Array(1, 3, 3)) should be(17)
    result1(Array(1, 3, 4)) should be(18)

    val result2 = t.mean(2)
    result2.size(1) should be(2)
    result2.size(2) should be(1)
    result2.size(3) should be(4)

    result2(Array(1, 1, 1)) should be(5)
    result2(Array(1, 1, 2)) should be(6)
    result2(Array(1, 1, 3)) should be(7)
    result2(Array(1, 1, 4)) should be(8)
    result2(Array(2, 1, 1)) should be(17)
    result2(Array(2, 1, 2)) should be(18)
    result2(Array(2, 1, 3)) should be(19)
    result2(Array(2, 1, 4)) should be(20)

    val result3 = t.mean(3)
    result3.size(1) should be(2)
    result3.size(2) should be(3)
    result3.size(3) should be(1)

    result3(Array(1, 1, 1)) should be(2.5)
    result3(Array(1, 2, 1)) should be(6.5)
    result3(Array(1, 3, 1)) should be(10.5)
    result3(Array(2, 1, 1)) should be(14.5)
    result3(Array(2, 2, 1)) should be(18.5)
    result3(Array(2, 3, 1)) should be(22.5)
  }

  "topk" should "be correct for 1d tensor" in {
    val t = Tensor(Storage(Array(0.0, 1.0, 5.0, 3.0, 9.0, 0.8, 6.3)))
    val (v, i) = t.topk(5)
    v should be(Tensor(Storage(Array(0.0, 0.8, 1.0, 3.0, 5.0))))
    i should be(Tensor(Storage(Array(1.0, 6.0, 2.0, 4.0, 3.0))))
  }

  "topk without increase" should "be correct for 1d tensor" in {
    val t = Tensor(Storage(Array(0.0, 1.0, 5.0, 3.0, 9.0, 0.8, 6.3)))
    val (v, i) = t.topk(5, increase = false)
    v should be(Tensor(Storage(Array(9.0, 6.3, 5.0, 3.0, 1.0))))
    i should be(Tensor(Storage(Array(5.0, 7.0, 3.0, 4.0, 2.0))))
  }

  "topk" should "be correct for 2d tensor" in {
    val t = Tensor(Storage(Array(
      0.0, 1.0, 5.0, 3.0, 9.0, 0.8, 6.3,
      0.0, 1.0, 5.0, 3.0, 9.0, 0.8, 6.3,
      0.0, 1.0, 5.0, 3.0, 9.0, 0.8, 6.3,
      0.0, 1.0, 5.0, 3.0, 9.0, 0.8, 6.3,
      0.0, 1.0, 5.0, 3.0, 9.0, 0.8, 6.3
    )), 1, Array(5, 7))
    val (v, i) = t.topk(5)
    v should be(Tensor(Storage(Array(
      0.0, 0.8, 1.0, 3.0, 5.0,
      0.0, 0.8, 1.0, 3.0, 5.0,
      0.0, 0.8, 1.0, 3.0, 5.0,
      0.0, 0.8, 1.0, 3.0, 5.0,
      0.0, 0.8, 1.0, 3.0, 5.0
    )), 1, Array(5, 5)))
    i should be(Tensor(Storage(Array(
      1.0, 6.0, 2.0, 4.0, 3.0,
      1.0, 6.0, 2.0, 4.0, 3.0,
      1.0, 6.0, 2.0, 4.0, 3.0,
      1.0, 6.0, 2.0, 4.0, 3.0,
      1.0, 6.0, 2.0, 4.0, 3.0
    )), 1, Array(5, 5)))
  }

  "topk without increase" should "be correct for 2d tensor" in {
    val t = Tensor(Storage(Array(
      0.0, 1.0, 5.0, 3.0, 9.0, 0.8, 6.3,
      0.0, 1.0, 5.0, 3.0, 9.0, 0.8, 6.3,
      0.0, 1.0, 5.0, 3.0, 9.0, 0.8, 6.3,
      0.0, 1.0, 5.0, 3.0, 9.0, 0.8, 6.3,
      0.0, 1.0, 5.0, 3.0, 9.0, 0.8, 6.3
    )), 1, Array(5, 7))
    val (v, i) = t.topk(5, increase = false)
    v should be(Tensor(Storage(Array(
      9.0, 6.3, 5.0, 3.0, 1.0,
      9.0, 6.3, 5.0, 3.0, 1.0,
      9.0, 6.3, 5.0, 3.0, 1.0,
      9.0, 6.3, 5.0, 3.0, 1.0,
      9.0, 6.3, 5.0, 3.0, 1.0
    )), 1, Array(5, 5)))
    i should be(Tensor(Storage(Array(
      5.0, 7.0, 3.0, 4.0, 2.0,
      5.0, 7.0, 3.0, 4.0, 2.0,
      5.0, 7.0, 3.0, 4.0, 2.0,
      5.0, 7.0, 3.0, 4.0, 2.0,
      5.0, 7.0, 3.0, 4.0, 2.0
    )), 1, Array(5, 5)))
  }

  "powx(x,a)" should "return correct value" in {
    val t: Tensor[Double] = Tensor(Storage(Array(2.0, 3.0, 4.0)))
    val r: Tensor[Double] = Tensor(Storage(Array(0.0, 0.0, 0.0)))
    r.pow(t, 2)
    r should be(Tensor(Storage(Array(4.0, 9.0, 16.0))))
  }

  "powx(a)" should "return correct value" in {
    val t: Tensor[Double] = Tensor(Storage(Array(2.0, 3.0, 4.0)))
    t.pow(2)
    t should be(Tensor(Storage(Array(4.0, 9.0, 16.0))))
  }

  "log(x)" should "return correct value" in {
    val t: Tensor[Double] = Tensor(Storage(Array(2.0, 3.0, 4.0)))
    val r: Tensor[Double] = Tensor(Storage(Array(0.0, 0.0, 0.0)))
    r.log(t)
    r should be(Tensor(Storage(Array(0.6931472, 1.0986123, 1.3862944))))
  }

  "log()" should "return correct value" in {
    val t: Tensor[Double] = Tensor(Storage(Array(2.0, 3.0, 4.0)))
    t.log(t)
    t should be(Tensor(Storage(Array(0.6931472, 1.0986123, 1.3862944))))
  }

  "exp(x)" should "return correct value" in {
    val t: Tensor[Double] = Tensor(Storage(Array(2.0, 3.0, 4.0)))
    val r: Tensor[Double] = Tensor(Storage(Array(0.0, 0.0, 0.0)))
    r.exp(t)
    r should be(Tensor(Storage(Array(7.389056, 20.085537, 54.59815))))
  }

  "exp()" should "return correct value" in {
    val t: Tensor[Double] = Tensor(Storage(Array(2.0, 3.0, 4.0)))
    t.exp()
    t should be(Tensor(Storage(Array(7.389056, 20.085537, 54.59815))))
  }

  "sqrt(x)" should "return correct value" in {
    val t: Tensor[Double] = Tensor(Storage(Array(2.0, 3.0, 4.0)))
    val r: Tensor[Double] = Tensor(Storage(Array(0.0, 0.0, 0.0)))
    r.sqrt(t)
    r should be(Tensor(Storage(Array(1.4142135, 1.7320508, 2.0))))
  }

  "sqrt()" should "return correct value" in {
    val t: Tensor[Double] = Tensor(Storage(Array(2.0, 3.0, 4.0)))
    t.sqrt()
    t should be(Tensor(Storage(Array(1.4142135, 1.7320508, 2.0))))
  }

  "log1p(x)" should "return correct value" in {
    val t: Tensor[Double] = Tensor(Storage(Array(2.0, 3.0, 4.0)))
    val r: Tensor[Double] = Tensor(Storage(Array(0.0, 0.0, 0.0)))
    r.log1p(t)
    r should be(Tensor(Storage(Array(1.0986123, 1.3862944, 1.609438))))
  }

  "log1p()" should "return correct value" in {
    val t: Tensor[Double] = Tensor(Storage(Array(2.0, 3.0, 4.0)))
    t.log1p()
    t should be(Tensor(Storage(Array(1.0986123, 1.3862944, 1.609438))))
  }

  "matrix sub(T)" should "return correct value" in{
    val a : Tensor[Double] = Tensor(Storage(Array(2.0, 3.0, 4.0)))
    val m = 1

    a.sub(m)

    a should be (Tensor(Storage(Array(1.0, 2.0, 3.0))))
  }

  "matrix sub(T,Tensor[T])" should "return correct value" in{
    val a : Tensor[Double] = Tensor(Storage(Array(2.0, 3.0, 4.0)))
    val b : Tensor[Double] = Tensor(Storage(Array(1.0, 2.0, 3.0)))
    val m = 2

    a.sub(m, b)
    a should be (Tensor(Storage(Array(0.0, -1.0, -2.0))))
  }

  "matrix sub(Tensor[T])" should "return correct value" in{
    val a : Tensor[Double] = Tensor(Storage(Array(2.0, 3.0, 4.0)))
    val b : Tensor[Double] = Tensor(Storage(Array(1.0, 2.0, 3.0)))

    a.sub(b)

    val r = Tensor(Storage(Array(1.0, 1.0, 1.0)))

    a should be (r)
  }

  "matrix sub(Tensor[T],T,Tensor[T])" should "return correct value" in{
    val a : Tensor[Double] = Tensor(Storage(Array(2.0, 3.0, 4.0)))
    val b : Tensor[Double] = Tensor(Storage(Array(1.0, 2.0, 3.0)))
    val c : Tensor[Double] = Tensor(Storage(Array(1.0, 2.0, 3.0)))

    val m = 2
    val d = a.sub(c, m, b)

    d should be (Tensor(Storage(Array(-1.0, -2.0, -3.0))))
  }

  "gemm(N, N)" should "return correct value" in {
    val matrixA = Tensor[Float](2, 3)
    val matrixB = Tensor[Float](3, 2)

    var i = 0
    matrixA.apply1(_ => {
      i = i + 1;
      i
    })
    matrixB.copy(matrixA)

    val matrixC = Tensor[Float](2, 2)

    DenseTensorBLAS.gemm[Float](
      'N', 'N',
      2, 2, 3,
      1,
      matrixA.storage().array(), matrixA.storageOffset() - 1, 2,
      matrixB.storage().array(), matrixB.storageOffset() - 1, 3,
      0,
      matrixC.storage().array(), matrixC.storageOffset() - 1, 2
    )

    val result = Tensor[Float](Storage(Array[Float](22, 28, 49, 64)), 1, Array(2, 2))

    matrixC should be (result)
  }

  "gemm(N, T)" should "return correct value" in {
    val matrixA = Tensor[Float](2, 3)
    val matrixB = Tensor[Float](2, 3)

    var i = 0
    matrixA.apply1(_ => {
      i = i + 1;
      i
    })
    matrixB.copy(matrixA)

    val matrixC = Tensor[Float](2, 2)

    DenseTensorBLAS.gemm[Float](
      'N', 'T',
      2, 2, 3,
      1,
      matrixA.storage().array(), matrixA.storageOffset() - 1, 2,
      matrixB.storage().array(), matrixB.storageOffset() - 1, 2,
      0,
      matrixC.storage().array(), matrixC.storageOffset() - 1, 2
    )

    val result = Tensor[Float](Storage(Array[Float](35, 44, 44, 56)), 1, Array(2, 2))
    matrixC should be (result)
  }

  "gemm(T, N)" should "return correct value" in {
    val matrixA = Tensor[Float](3, 2)
    val matrixB = Tensor[Float](3, 2)

    var i = 0
    matrixA.apply1(_ => {
      i = i + 1;
      i
    })
    matrixB.copy(matrixA)

    val matrixC = Tensor[Float](2, 2)

    DenseTensorBLAS.gemm[Float](
      't', 'n',
      2, 2, 3,
      1,
      matrixA.storage().array(), matrixA.storageOffset() - 1, 3,
      matrixB.storage().array(), matrixB.storageOffset() - 1, 3,
      0,
      matrixC.storage().array(), matrixC.storageOffset() - 1, 2
    )

    val result = Tensor[Float](Storage(Array[Float](14, 32, 32, 77)), 1, Array(2, 2))
    matrixC should be (result)
  }

  "gemm(T, T)" should "return correct value" in {
    val matrixA = Tensor[Float](3, 2)
    val matrixB = Tensor[Float](2, 3)

    var i = 0
    matrixA.apply1(_ => {
      i = i + 1;
      i
    })
    matrixB.copy(matrixA)

    val matrixC = Tensor[Float](2, 2)

    DenseTensorBLAS.gemm[Float](
      'T', 'T',
      2, 2, 3,
      1,
      matrixA.storage().array(), matrixA.storageOffset() - 1, 3,
      matrixB.storage().array(), matrixB.storageOffset() - 1, 2,
      0,
      matrixC.storage().array(), matrixC.storageOffset() - 1, 2
    )

    val result = Tensor[Float](Storage(Array[Float](22, 49, 28, 64)), 1, Array(2, 2))

    matrixC should be (result)
  }

  "cdiv" should "return right result" in {
    val x = Tensor[Float](2, 2).fill(1f)
    val y = Tensor(Storage(Array(1f, 2, 3, 4)), 1, Array(2, 2))

    x.cdiv(y)

    x should be (Tensor(Storage(Array(1f / 1, 1f / 2, 1f / 3, 1f / 4)), 1, Array(2, 2)))
    y should be (Tensor(Storage(Array(1f, 2, 3, 4)), 1, Array(2, 2)))
  }

  "cdiv" should "return right result 2" in {
    val x = Tensor[Float](2, 2).fill(1f)
    val y = Tensor(Storage(Array(1f, 2, 3, 4)), 1, Array(2, 2))

    y.cdiv(x, y)

    x should be (Tensor(Storage(Array(1f, 1f, 1f, 1f)), 1, Array(2, 2)))
    y should be (Tensor(Storage(Array(1f / 1, 1f / 2, 1f / 3, 1f / 4)), 1, Array(2, 2)))
  }

  "cdiv" should "return right result 3" in {
    val x = Tensor[Float](2, 2).fill(1f)
    val y = Tensor(Storage(Array(1f, 2, 3, 4)), 1, Array(2, 2))
    val z = Tensor[Float](2, 2).zero()

    z.cdiv(x, y)

    x should be (Tensor(Storage(Array(1f, 1f, 1f, 1f)), 1, Array(2, 2)))
    y should be (Tensor(Storage(Array(1f, 2, 3, 4)), 1, Array(2, 2)))
    z should be (Tensor(Storage(Array(1f / 1, 1f / 2, 1f / 3, 1f / 4)), 1, Array(2, 2)))
  }

  "cmul" should "return right result" in {
    val x = Tensor[Float](2, 2).fill(2f)
    val y = Tensor(Storage(Array(1f, 2, 3, 4)), 1, Array(2, 2))

    x.cmul(y)

    x should be (Tensor(Storage(Array(2f * 1, 2f * 2, 2f * 3, 2f * 4)), 1, Array(2, 2)))
    y should be (Tensor(Storage(Array(1f, 2, 3, 4)), 1, Array(2, 2)))
  }

  "cmul" should "return right result 2" in {
    val x = Tensor[Float](2, 2).fill(2f)
    val y = Tensor(Storage(Array(1f, 2, 3, 4)), 1, Array(2, 2))

    y.cmul(x, y)

    x should be (Tensor(Storage(Array(2f, 2f, 2f, 2f)), 1, Array(2, 2)))
    y should be (Tensor(Storage(Array(2f * 1, 2f * 2, 2f * 3, 2f * 4)), 1, Array(2, 2)))
  }

  "cmul" should "return right result 3" in {
    val x = Tensor[Float](2, 2).fill(2f)
    val y = Tensor(Storage(Array(1f, 2, 3, 4)), 1, Array(2, 2))
    val z = Tensor[Float](2, 2).zero()

    z.cmul(x, y)

    x should be (Tensor(Storage(Array(2f, 2f, 2f, 2f)), 1, Array(2, 2)))
    y should be (Tensor(Storage(Array(1f, 2, 3, 4)), 1, Array(2, 2)))
    z should be (Tensor(Storage(Array(2f * 1, 2f * 2, 2f * 3, 2f * 4)), 1, Array(2, 2)))
  }

  "cmul" should "return right result 4" in {
    val x = Tensor[Float](Storage(Array(1f, 2)), 1, Array(2, 1))
    val y = Tensor(Storage(Array(1f, 2, 3, 4, 5, 6)), 1, Array(2, 3))
    x.expandAs(y)
    val z = Tensor[Float](2, 3).zero()

    z.cmul(x, y)

    x should be (Tensor(Storage(Array(1f, 2)), 1, Array(2, 3), Array(1, 0)))
    y should be (Tensor(Storage(Array(1f, 2, 3, 4, 5, 6)), 1, Array(2, 3)))
    z should be (Tensor(Storage(Array(1f * 1, 1f * 2, 1f * 3, 2f * 4, 2f * 5, 2f * 6)),
      1, Array(2, 3)))
  }

  "cmul" should "return right result 5" in {
    val x = Tensor[Float](Storage(Array(1f, 2, 3)), 1, Array(1, 3))
    val y = Tensor(Storage(Array(1f, 2, 3, 4, 5, 6)), 1, Array(2, 3))
    x.expandAs(y)
    val z = Tensor[Float](2, 3).zero()

    z.cmul(x, y)

    x should be (Tensor(Storage(Array(1f, 2, 3)), 1, Array(2, 3), Array(0, 1)))
    y should be (Tensor(Storage(Array(1f, 2, 3, 4, 5, 6)), 1, Array(2, 3)))
    z should be (Tensor(Storage(Array(1f * 1, 2f * 2, 3f * 3, 1f * 4, 2f * 5, 3f * 6)),
      1, Array(2, 3)))
  }

  "add" should "return right result" in {
    val x = Tensor[Float](2, 2).fill(2f)
    val y = Tensor(Storage(Array(1f, 2, 3, 4)), 1, Array(2, 2))

    x.add(y)

    x should be (Tensor(Storage(Array(2f + 1, 2f + 2, 2f + 3, 2f + 4)), 1, Array(2, 2)))
    y should be (Tensor(Storage(Array(1f, 2, 3, 4)), 1, Array(2, 2)))
  }

  "add" should "return right result 2" in {
    val x = Tensor[Float](2, 2).fill(2f)
    val y = Tensor(Storage(Array(1f, 2, 3, 4)), 1, Array(2, 2))

    y.add(x, 2, y)

    x should be (Tensor(Storage(Array(2f, 2f, 2f, 2f)), 1, Array(2, 2)))
    y should be (Tensor(Storage(Array(2f + 2, 2f + 4, 2f + 6, 2f + 8)), 1, Array(2, 2)))
  }

  "addmv on 1 element vector" should "return right result" in {
    val mat = Tensor[Float](84, 1).fill(2.0f)
    val vec = Tensor[Float](1).fill(3.0f)

    val r = Tensor[Float](84).fill(9.0f)

    DenseTensorMath.addmv[Float](r, 1.0f, r, 1.0f, mat, vec)

    r should be (Tensor[Float](84).fill(15f))
  }

  "addmv on 1 element vector" should "return right result 1" in {
    val mat = Tensor[Float](84, 1).fill(2.0f)
    val vec = Tensor[Float](2).narrow(1, 2, 1).fill(3.0f)

    val r = Tensor[Float](84).fill(9.0f)

    DenseTensorMath.addmv[Float](r, 1.0f, r, 1.0f, mat, vec)

    r should be (Tensor[Float](84).fill(15f))
  }

  "addmv on 1 element vector" should "return right result 2" in {
    val mat = Tensor[Float](84, 2).narrow(2, 1, 1).fill(2.0f)
    val vec = Tensor[Float](2).narrow(1, 1, 1).fill(3.0f)

    val r = Tensor[Float](84).fill(9.0f)

    DenseTensorMath.addmv[Float](r, 1.0f, r, 1.0f, mat, vec)

    r should be (Tensor[Float](84).fill(15f))
  }

  "addmv on 1 element vector" should "return right result 3" in {
    val mat = Tensor[Float](1, 3).fill(2.0f).t
    val vec = Tensor[Float](1).fill(3.0f)

    val r = Tensor[Float](3).fill(9.0f)

    DenseTensorMath.addmv[Float](r, 0.5f, r, 2.0f, mat, vec)

    r should be (Tensor[Float](3).fill(16.5f))
  }

  "dot on 1D vector" should "return right result 70" in {
    val x = Tensor[Float](Storage(Array[Float](1, 2, 3, 4)))
    val y = Tensor[Float](Storage(Array[Float](5, 6, 7, 8)))
    val s = x.dot(y)

    s should be (70)
  }

  "dot on 2D vector" should "return right result 70" in {
    val x = Tensor[Float](Storage(Array[Float](1, 2, 3, 4)), 1, Array(2, 2))
    val y = Tensor[Float](Storage(Array[Float](5, 6, 7, 8)), 1, Array(2, 2))
    val s = x.dot(y)

    s should be (70)
  }
}
