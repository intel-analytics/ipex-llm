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
class TensorNumericSpec extends FlatSpec with Matchers {
  "Int Tensor" should "works correctly" in {
    val a = Tensor[Int](T(1, 2, 3))
    val b = Tensor[Int](T(2, 3, 4))
    val sum = Tensor[Int](T(3, 5, 7))

    a + b equals sum
  }

  "Short Tensor" should "works correctly" in {
    val a = Tensor[Short](T(1.toShort, 2.toShort, 3.toShort))
    val b = Tensor[Short](T(2.toShort, 3.toShort, 4.toShort))
    val sum = Tensor[Short](T(3.toShort, 5.toShort, 7.toShort))

    a + b equals sum
  }

  "Long Tensor" should "works correctly" in {
    val a = Tensor[Long](T(1L, 2L, 3L))
    val b = Tensor[Long](T(2L, 3L, 4L))
    val sum = Tensor[Long](T(3L, 5L, 7L))

    a + b equals sum
  }

  "Char Tensor" should "works correctly" in {
    val a = Tensor[Char](T('a', 'b', 'c'))
    val b = Tensor[Char](T('a', 'b', 'c'))
    val sum = Tensor[Char](T('Â', 'Ä', 'Æ'))

    a + b equals sum
  }

  "String Tensor" should "works correctly" in {
    val a = Tensor[String](T("a", "b", "c"))
    val b = Tensor[String](T("a", "b", "c"))
    val sum = Tensor[String](T("aa", "bb", "cc"))

    a + b equals sum
  }

  "Boolean Tensor" should "works correctly" in {
    val a = Tensor[Boolean](T(true, false, false))
    val b = Tensor[Boolean](T(true, true, false))
    val sum = Tensor[Boolean](T(true, false, false))

    import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

    a.map(b,
      (x, y) => TensorNumeric.NumericBoolean.and(x, y))
  }
}
