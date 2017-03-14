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

package com.intel.analytics.bigdl.parameters

import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Serial
class FP16SplitsParameterSpec extends FlatSpec with Matchers {
  "convert double tensor to fp16 array and back" should "be same when the number is integer" in {
    val tensor = Tensor[Double](5)
    tensor.setValue(1, 1.0)
    tensor.setValue(2, 2.0)
    tensor.setValue(3, 3.0)
    tensor.setValue(4, 4.0)
    tensor.setValue(5, 5.0)

    val target = tensor.clone().zero()

    val parameter = new FP16SplitsCompressedTensor[Double](tensor, 3)
    parameter.deCompress(target)

    target should be(tensor)
  }

  it should "get a truncated value when the number is float with to many mantissa" in {
    val tensor = Tensor[Double](5)
    tensor.setValue(1, 1.111111)
    tensor.setValue(2, 2.111111)
    tensor.setValue(3, 3.111111)
    tensor.setValue(4, 4.111111)
    tensor.setValue(5, 5.111111)

    val parameter = new FP16SplitsCompressedTensor[Double](tensor, 3)
    parameter.deCompress(tensor)

    val target = Tensor[Double](5)
    target.setValue(1, 1.109375)
    target.setValue(2, 2.109375)
    target.setValue(3, 3.109375)
    target.setValue(4, 4.09375)
    target.setValue(5, 5.09375)

    tensor should be(target)
  }

  it should "be correct when only perform on a slice" in {
    val tensor = Tensor[Double](5)
    tensor.setValue(1, 1.111111)
    tensor.setValue(2, 2.111111)
    tensor.setValue(3, 3.111111)
    tensor.setValue(4, 4.111111)
    tensor.setValue(5, 5.111111)

    val parameter = new FP16SplitsCompressedTensor[Double](tensor, 3)
    val buffer = parameter.bytes(2, 2)
    val parameter2 = new FP16CompressedTensor[Double](buffer)
    val test = tensor.clone()
    parameter2.deCompress(0, test, 2, 2)

    val target = Tensor[Double](5)
    target.setValue(1, 1.111111)
    target.setValue(2, 2.111111)
    target.setValue(3, 3.109375)
    target.setValue(4, 4.09375)
    target.setValue(5, 5.111111)

    test should be(target)
  }

  it should "throw exception when slice size is not consisted" in {
    val tensor = Tensor[Double](5)
    val parameter = new FP16SplitsCompressedTensor[Double](tensor, 3)
    intercept[IllegalArgumentException] {
      parameter.bytes(1, 2)
    }
  }

  "convert float tensor to fp16 array and back" should "be same when the number is integer" in {
    val tensor = Tensor[Float](5)
    tensor.setValue(1, 1.0f)
    tensor.setValue(2, 2.0f)
    tensor.setValue(3, 3.0f)
    tensor.setValue(4, 4.0f)
    tensor.setValue(5, 5.0f)

    val target = tensor.clone().zero()

    val parameter = new FP16SplitsCompressedTensor[Float](tensor, 3)
    parameter.deCompress(target)

    target should be(tensor)
  }


  it should "get a truncated value when the number is float with to many mantissa" in {
    val tensor = Tensor[Float](5)
    tensor.setValue(1, 1.111111f)
    tensor.setValue(2, 2.111111f)
    tensor.setValue(3, 3.111111f)
    tensor.setValue(4, 4.111111f)
    tensor.setValue(5, 5.111111f)

    val parameter = new FP16SplitsCompressedTensor[Float](tensor, 3)
    parameter.deCompress(tensor)

    val target = Tensor[Float](5)
    target.setValue(1, 1.109375f)
    target.setValue(2, 2.109375f)
    target.setValue(3, 3.109375f)
    target.setValue(4, 4.09375f)
    target.setValue(5, 5.09375f)

    tensor should be(target)
  }

  it should "be correct when only perform on a slice" in {
    val tensor = Tensor[Float](5)
    tensor.setValue(1, 1.111111f)
    tensor.setValue(2, 2.111111f)
    tensor.setValue(3, 3.111111f)
    tensor.setValue(4, 4.111111f)
    tensor.setValue(5, 5.111111f)

    val parameter = new FP16SplitsCompressedTensor[Float](tensor, 3)
    val buffer = parameter.bytes(2, 2)
    val parameter2 = new FP16CompressedTensor[Float](buffer)
    val test = tensor.clone()
    parameter2.deCompress(0, test, 2, 2)

    val target = Tensor[Float](5)
    target.setValue(1, 1.111111f)
    target.setValue(2, 2.111111f)
    target.setValue(3, 3.109375f)
    target.setValue(4, 4.09375f)
    target.setValue(5, 5.111111f)

    test should be(target)
  }

  it should "throw exception when slice size is not consisted" in {
    val tensor = Tensor[Float](5)
    val parameter = new FP16SplitsCompressedTensor[Float](tensor, 3)
    intercept[IllegalArgumentException] {
      parameter.bytes(1, 2)
    }
  }
}
