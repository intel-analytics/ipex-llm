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

package com.intel.analytics.bigdl.nn.quantization

import com.intel.analytics.bigdl.nn.Quantize._
import com.intel.analytics.bigdl.tensor.Tensor

import java.nio.ByteBuffer
import org.scalatest.{FlatSpec, Matchers}

class QuantizeSpec extends FlatSpec with Matchers {
  "Quantize a number with same sign max and min" should "generate correct output" in {
    val src = 0.6f
    val (max, min) = (0.7f, 0.2f)

    val dst = quantize(src, max, min)
    dst should be (109)

    val result = dequantize(dst, max, min)
    result should be (0.6f +- 0.01f)

    val los = Math.abs(result - src).toDouble
    los should be (0.toDouble +- 0.01) // here is an experiment result
  }

  "Quantize a number with different sign max and min" should "generate correct output" in {
    val src = -0.6f
    val (max, min) = (0.7f, -0.2f)

    val dst = quantize(src, max, min)
    dst should be (-109)

    val result = dequantize(dst, max, min)
    result should be (-0.6f +- 0.01f)

    val los = Math.abs(result - src).toDouble
    los should be (0.toDouble +- 0.01) // here is an experiment result
  }

  "Quantize a array" should "generate correct output" in {
    val src = Array[Float](0.6f, 0.4f, -0.3f, 0.2f, 0.1f)

    val dst = ByteBuffer.allocate(src.length)
    dst.clear()

    val (max, min) = quantize(src, 0, src.length, dst, 0)

    dst.get(0) should be (127)
    dst.get(1) should be (85)
    dst.get(2) should be (-63)
    dst.get(3) should be (42)
    dst.get(4) should be (21)

    val before = src.clone()
    for (i <- src.indices) {
      src(i) = 0f
    }

    dequantize(src, 0, src.length, dst, 0, max, min)

    src(0) should be (0.6f +- 0.01f)
    src(1) should be (0.4f +- 0.01f)
    src(2) should be (-0.3f +- 0.01f)
    src(3) should be (0.2f +- 0.01f)
    src(4) should be (0.1f +- 0.01f)

    val after = src.clone()

    val los = loss(before, after, 0, src.length)

    los should be (0.toDouble +- 0.01) // here is an experiment result
  }

  "Quantize a matrix" should "generate correct output" in {
    val src = Array(0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.4f, 0.3f, 0.2f, 0.1f)
    val dst = ByteBuffer.allocate(src.length)
    dst.clear()

    val (max, min) = quantize(src, 0, src.length, dst, 0, Array(2, 5))

    for (i <- src.indices) {
      println(dst.get(i))
    }

    dst.get(0) should be (25)
    dst.get(1) should be (51)
    dst.get(2) should be (76)
    dst.get(3) should be (102)
    dst.get(4) should be (127)
    dst.get(5) should be (127)
    dst.get(6) should be (85)
    dst.get(7) should be (64)
    dst.get(8) should be (42)
    dst.get(9) should be (21)

    val before = src.clone()
    for (i <- src.indices) {
      src(i) = 0f
    }

    dequantize(src, 0, src.length, dst, 0, max, min, Array(2, 5))
    for (i <- src.indices) {
      println(src(i))
    }

    src(0) should be (0.1f +- 0.01f)
    src(1) should be (0.2f +- 0.01f)
    src(2) should be (0.3f +- 0.01f)
    src(3) should be (0.4f +- 0.01f)
    src(4) should be (0.5f +- 0.01f)
    src(5) should be (0.6f +- 0.01f)
    src(6) should be (0.4f +- 0.01f)
    src(7) should be (0.3f +- 0.01f)
    src(8) should be (0.2f +- 0.01f)
    src(9) should be (0.1f +- 0.01f)

    val after = src.clone()

    val los = loss(before, after, 0, src.length)
    los should be (0.toDouble +- 0.02) // here is an experiment result
  }

  "Quantize a 1-d tensor" should "generate correct output" in {
    val array = Array[Float](0.6f, 0.4f, -0.3f, 0.2f, 0.1f)
    val src = Tensor[Float](array, Array(5))

    val dst = ByteBuffer.allocate(src.nElement())
    dst.clear()

    val (max, min) = quantize(src, dst, 0)

    dst.get(0) should be (127)
    dst.get(1) should be (85)
    dst.get(2) should be (-63)
    dst.get(3) should be (42)
    dst.get(4) should be (21)

    val before = src.clone()
    src.apply1(_ => 0f)

    dequantize(src, dst, 0, max, min)

    src.valueAt(1) should be (0.6f +- 0.01f)
    src.valueAt(2) should be (0.4f +- 0.01f)
    src.valueAt(3) should be (-0.3f +- 0.01f)
    src.valueAt(4) should be (0.2f +- 0.01f)
    src.valueAt(5) should be (0.1f +- 0.01f)

    val after = src.clone()

    val los = loss(before.storage().array(), after.storage().array(), 0, src.nElement())
    los should be (0.toDouble +- 0.01) // here is an experiment result
  }

  "Quantize a 2-d tensor" should "generate correct output" in {
    val array = Array(0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.4f, 0.3f, 0.2f, 0.1f)
    val src = Tensor[Float](array, Array(2, 5))

    val dst = ByteBuffer.allocate(src.nElement())
    dst.clear()

    val (max, min) = quantize(src, dst, 0)

    dst.get(0) should be (25)
    dst.get(1) should be (51)
    dst.get(2) should be (76)
    dst.get(3) should be (102)
    dst.get(4) should be (127)
    dst.get(5) should be (127)
    dst.get(6) should be (85)
    dst.get(7) should be (64)
    dst.get(8) should be (42)
    dst.get(9) should be (21)

    val before = src.clone()
    src.apply1(_ => 0f)

    dequantize(src, dst, 0, max, min)

    src.valueAt(1, 1) should be (0.1f +- 0.01f)
    src.valueAt(1, 2) should be (0.2f +- 0.01f)
    src.valueAt(1, 3) should be (0.3f +- 0.01f)
    src.valueAt(1, 4) should be (0.4f +- 0.01f)
    src.valueAt(1, 5) should be (0.5f +- 0.01f)
    src.valueAt(2, 1) should be (0.6f +- 0.01f)
    src.valueAt(2, 2) should be (0.4f +- 0.01f)
    src.valueAt(2, 3) should be (0.3f +- 0.01f)
    src.valueAt(2, 4) should be (0.2f +- 0.01f)
    src.valueAt(2, 5) should be (0.1f +- 0.01f)

    val after = src.clone()

    val los = loss(before.storage().array(), after.storage().array(), 0, src.nElement())
    los should be (0.toDouble +- 0.02) // here is an experiment result
  }

  "Quantize a 3-d tensor" should "generate correct output" in {
    val array = Array(
      0.1f, 0.2f, 0.3f,
      0.4f, 0.5f, 0.6f,

      -0.5f, 0.4f, 0.3f,
      0.2f, 0.1f, 0f
    )
    val src = Tensor[Float](array, Array(2, 2, 3))

    val dst = ByteBuffer.allocate(src.nElement())
    dst.clear()

    val (max, min) = quantize(src, dst, 0)

    dst.get(0) should be (21)
    dst.get(1) should be (42)
    dst.get(2) should be (64)
    dst.get(3) should be (85)
    dst.get(4) should be (106)
    dst.get(5) should be (127)
    dst.get(6) should be (-127)
    dst.get(7) should be (102)
    dst.get(8) should be (76)
    dst.get(9) should be (51)
    dst.get(10) should be (25)
    dst.get(11) should be (0)

    val before = src.clone()
    src.apply1(_ => 0f)

    dequantize(src, dst, 0, max, min)

    src.valueAt(1, 1, 1) should be (0.1f +- 0.01f)
    src.valueAt(1, 1, 2) should be (0.2f +- 0.01f)
    src.valueAt(1, 1, 3) should be (0.3f +- 0.01f)
    src.valueAt(1, 2, 1) should be (0.4f +- 0.01f)
    src.valueAt(1, 2, 2) should be (0.5f +- 0.01f)
    src.valueAt(1, 2, 3) should be (0.6f +- 0.01f)
    src.valueAt(2, 1, 1) should be (-0.5f +- 0.01f)
    src.valueAt(2, 1, 2) should be (0.4f +- 0.01f)
    src.valueAt(2, 1, 3) should be (0.3f +- 0.01f)
    src.valueAt(2, 2, 1) should be (0.2f +- 0.01f)
    src.valueAt(2, 2, 2) should be (0.1f +- 0.01f)
    src.valueAt(2, 2, 3) should be (0.0f +- 0.01f)

    val after = src.clone()

    val los = loss(before.storage().array(), after.storage().array(), 0, src.nElement())
    los should be (0.toDouble +- 0.02) // here is an experiment result
  }
}
