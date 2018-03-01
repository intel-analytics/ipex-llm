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

import com.google.protobuf.ByteString
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}

class NotEqualSpec extends FlatSpec with Matchers {
  "NotEqual Float operation" should "works correctly" in {
    val input =
      T(
        Tensor[Float](T(1f, 2f, 2f)),
        Tensor[Float](T(2f, 3f, 2f))
      )

    val expectOutput = Tensor[Boolean](T(true, true, false))

    val output = NotEqual[Boolean]().forward(input)
    output should be(expectOutput)
  }

  "NotEqual Boolean operation" should "works correctly" in {
    val input =
      T(
        Tensor[Boolean](T(true, true, false)),
        Tensor[Boolean](T(false, true, false))
      )

    val expectOutput = Tensor[Boolean](T(true, false, false))

    val output = NotEqual[Boolean]().forward(input)
    output should be(expectOutput)
  }

  "NotEqual Double operation" should "works correctly" in {
    val input =
      T(
        Tensor[Double](T(1.0, 2.0, 2.0)),
        Tensor[Double](T(2.0, 3.0, 2.0))
      )

    val expectOutput = Tensor[Boolean](T(true, true, false))

    val output = NotEqual[Boolean]().forward(input)
    output should be(expectOutput)
  }

  "NotEqual Char operation" should "works correctly" in {
    val input =
      T(
        Tensor[Char](T('a', 'b', 'a')),
        Tensor[Char](T('b', 'c', 'a'))
      )

    val expectOutput = Tensor[Boolean](T(true, true, false))

    val output = NotEqual[Boolean]().forward(input)
    output should be(expectOutput)
  }

  "NotEqual Long operation" should "works correctly" in {
    val input =
      T(
        Tensor[Long](T(1L, 2L, 2L)),
        Tensor[Long](T(2L, 3L, 2L))
      )

    val expectOutput = Tensor[Boolean](T(true, true, false))

    val output = NotEqual[Boolean]().forward(input)
    output should be(expectOutput)
  }

  "NotEqual String operation" should "works correctly" in {
    import com.intel.analytics.bigdl.utils.tf.TFTensorNumeric.NumericByteString
    val input =
      T(

        Tensor[ByteString](Array(ByteString.copyFromUtf8("abc"),
          ByteString.copyFromUtf8("bbb"),
          ByteString.copyFromUtf8("aaaa").substring(0, 3)), Array(3)),
        Tensor[ByteString](Array(ByteString.copyFromUtf8("aaa"),
          ByteString.copyFromUtf8("ccc"),
          ByteString.copyFromUtf8("aaa")), Array(3))
      )

    val expectOutput = Tensor[Boolean](T(true, true, false))

    val output = NotEqual[Boolean]().forward(input)
    output should be(expectOutput)
  }

  "NotEqual Short operation" should "works correctly" in {
    val input =
      T(
        Tensor[Short](T(1: Short, 2: Short, 2: Short)),
        Tensor[Short](T(2: Short, 3: Short, 2: Short))
      )

    val expectOutput = Tensor[Boolean](T(true, true, false))

    val output = NotEqual[Boolean]().forward(input)
    output should be(expectOutput)
  }

  "NotEqual Int operation" should "works correctly" in {
    val input =
      T(
        Tensor[Int](T(1, 2, 2)),
        Tensor[Int](T(2, 3, 2))
      )

    val expectOutput = Tensor[Boolean](T(true, true, false))

    val output = NotEqual[Boolean]().forward(input)
    output should be(expectOutput)
  }
}

class NotEqualSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val notEqual = NotEqual[Float].setName("notEqual")
    val input = T(Tensor[Boolean](T(true, false)), Tensor[Boolean](T(true, false)))
    runSerializationTest(notEqual, input, notEqual
      .asInstanceOf[ModuleToOperation[Float]].module.getClass)
  }
}
