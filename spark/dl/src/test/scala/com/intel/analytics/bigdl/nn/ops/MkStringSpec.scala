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
package com.intel.analytics.bigdl.nn.ops

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}

class MkStringSpec extends FlatSpec with Matchers {
  "MkString Operation with DenseTensor" should "work correctly" in {
    val input = Tensor[Double](
      T(T(1.0, 2.0, 3.0),
        T(4.0, 5.0, 6.0)))

    val expectOutput = Tensor[String](T("1.0,2.0,3.0", "4.0,5.0,6.0"))

    val output = MkString[Double]().forward(input)
    output should be(expectOutput)
  }
  "MkString Operation with SparseTensor" should "work correctly" in {
    val input = Tensor.sparse(
      indices = Array(Array(0, 0, 1, 1, 1, 2), Array(0, 1, 0, 1, 2, 2)),
      values = Array(1, 2, 3, 4, 5, 6),
      shape = Array(3, 4)
    )

    val expectOutput = Tensor[String](T("1,2", "3,4,5", "6"))

    val output = MkString[Double]().forward(input)
    output should be(expectOutput)
  }
}

class MkStringSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val mkString = new MkString[Float](strDelimiter = ",").setName("MkString")
    val input = Tensor.sparse(
      indices = Array(Array(0, 0, 1, 1, 1, 2), Array(0, 1, 0, 1, 2, 2)),
      values = Array(1, 2, 3, 4, 5, 6),
      shape = Array(3, 4)
    )
    runSerializationTest(mkString, input)
  }
}
