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

@com.intel.analytics.bigdl.tags.Parallel
class TensorConvSpec extends FlatSpec with Matchers {
  "Valid conv" should "return correct value" in {
    val t = new DenseTensor[Double](3, 4)
    t(Array(1, 1)) = 1
    t(Array(1, 2)) = 2
    t(Array(1, 3)) = 3
    t(Array(1, 4)) = 4
    t(Array(2, 1)) = 2
    t(Array(2, 2)) = 3
    t(Array(2, 3)) = 4
    t(Array(2, 4)) = 5
    t(Array(3, 1)) = 3
    t(Array(3, 2)) = 4
    t(Array(3, 3)) = 5
    t(Array(3, 4)) = 6

    val k = new DenseTensor[Double](2, 2)
    k(Array(1, 1)) = 1
    k(Array(1, 2)) = 2
    k(Array(2, 1)) = 3
    k(Array(2, 2)) = 4

    val r = t.conv2(k)
    r.size(1) should be(2)
    r.size(2) should be(3)
    r(Array(1, 1)) should be(17)
    r(Array(1, 2)) should be(27)
    r(Array(1, 3)) should be(37)
    r(Array(2, 1)) should be(27)
    r(Array(2, 2)) should be(37)
    r(Array(2, 3)) should be(47)
  }

  "Full conv" should "return correct value" in {
    val t = new DenseTensor[Double](3, 4)
    t(Array(1, 1)) = 1
    t(Array(1, 2)) = 2
    t(Array(1, 3)) = 3
    t(Array(1, 4)) = 4
    t(Array(2, 1)) = 2
    t(Array(2, 2)) = 3
    t(Array(2, 3)) = 4
    t(Array(2, 4)) = 5
    t(Array(3, 1)) = 3
    t(Array(3, 2)) = 4
    t(Array(3, 3)) = 5
    t(Array(3, 4)) = 6

    val k = new DenseTensor[Double](2, 2)
    k(Array(1, 1)) = 1
    k(Array(1, 2)) = 2
    k(Array(2, 1)) = 3
    k(Array(2, 2)) = 4

    val r = t.conv2(k, 'F')
    r.size(1) should be(4)
    r.size(2) should be(5)
    r(Array(1, 1)) should be(1)
    r(Array(1, 2)) should be(4)
    r(Array(1, 3)) should be(7)
    r(Array(1, 4)) should be(10)
    r(Array(1, 5)) should be(8)
    r(Array(2, 1)) should be(5)
    r(Array(2, 2)) should be(17)
    r(Array(2, 3)) should be(27)
    r(Array(2, 4)) should be(37)
    r(Array(2, 5)) should be(26)
    r(Array(3, 1)) should be(9)
    r(Array(3, 2)) should be(27)
    r(Array(3, 3)) should be(37)
    r(Array(3, 4)) should be(47)
    r(Array(3, 5)) should be(32)
    r(Array(4, 1)) should be(9)
    r(Array(4, 2)) should be(24)
    r(Array(4, 3)) should be(31)
    r(Array(4, 4)) should be(38)
    r(Array(4, 5)) should be(24)
  }

  "Valid xcorr" should "return correct value" in {
    val t = new DenseTensor[Double](3, 4)
    t(Array(1, 1)) = 1
    t(Array(1, 2)) = 2
    t(Array(1, 3)) = 3
    t(Array(1, 4)) = 4
    t(Array(2, 1)) = 2
    t(Array(2, 2)) = 3
    t(Array(2, 3)) = 4
    t(Array(2, 4)) = 5
    t(Array(3, 1)) = 3
    t(Array(3, 2)) = 4
    t(Array(3, 3)) = 5
    t(Array(3, 4)) = 6

    val k = new DenseTensor[Double](2, 2)
    k(Array(1, 1)) = 1
    k(Array(1, 2)) = 2
    k(Array(2, 1)) = 3
    k(Array(2, 2)) = 4

    val r = t.xcorr2(k)
    r.size(1) should be(2)
    r.size(2) should be(3)
    r(Array(1, 1)) should be(23)
    r(Array(1, 2)) should be(33)
    r(Array(1, 3)) should be(43)
    r(Array(2, 1)) should be(33)
    r(Array(2, 2)) should be(43)
    r(Array(2, 3)) should be(53)
  }

  "Full Xcorr" should "return correct value" in {
    val t = new DenseTensor[Double](3, 4)
    t(Array(1, 1)) = 1
    t(Array(1, 2)) = 2
    t(Array(1, 3)) = 3
    t(Array(1, 4)) = 4
    t(Array(2, 1)) = 2
    t(Array(2, 2)) = 3
    t(Array(2, 3)) = 4
    t(Array(2, 4)) = 5
    t(Array(3, 1)) = 3
    t(Array(3, 2)) = 4
    t(Array(3, 3)) = 5
    t(Array(3, 4)) = 6

    val k = new DenseTensor[Double](2, 2)
    k(Array(1, 1)) = 1
    k(Array(1, 2)) = 2
    k(Array(2, 1)) = 3
    k(Array(2, 2)) = 4

    val r = t.xcorr2(k, 'F')
    r.size(1) should be(4)
    r.size(2) should be(5)
    r(Array(1, 1)) should be(4)
    r(Array(1, 2)) should be(11)
    r(Array(1, 3)) should be(18)
    r(Array(1, 4)) should be(25)
    r(Array(1, 5)) should be(12)
    r(Array(2, 1)) should be(10)
    r(Array(2, 2)) should be(23)
    r(Array(2, 3)) should be(33)
    r(Array(2, 4)) should be(43)
    r(Array(2, 5)) should be(19)
    r(Array(3, 1)) should be(16)
    r(Array(3, 2)) should be(33)
    r(Array(3, 3)) should be(43)
    r(Array(3, 4)) should be(53)
    r(Array(3, 5)) should be(23)
    r(Array(4, 1)) should be(6)
    r(Array(4, 2)) should be(11)
    r(Array(4, 3)) should be(14)
    r(Array(4, 4)) should be(17)
    r(Array(4, 5)) should be(6)
  }
}
