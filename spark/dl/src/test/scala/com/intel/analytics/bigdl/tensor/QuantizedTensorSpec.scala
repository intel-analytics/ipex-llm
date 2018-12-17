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

import com.intel.analytics.bigdl.bigquant.BigQuant
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.nn.quantized.{LinearWeight, LinearWeightParams}
import org.apache.commons.lang.SerializationUtils
import org.scalatest.{FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Parallel
class QuantizedTensorSpec extends FlatSpec with Matchers {
  "A QuantizeTensor set to empty" should "work correctly" in {
    val inputChannel = 4
    val outputChannel = 4

    val fp32Tensor = Tensor(outputChannel, inputChannel).rand()
    val tensor = QuantizedTensor[Float](fp32Tensor, LinearWeightParams(outputChannel, inputChannel))

    tensor.set()

    tensor.getNativeStorage should be (0L)
    tensor.getStorage should be (null)

    tensor.release()
  }

  "A QuantizeTensor set to other tensor" should "work correctly" in {
    val inputChannel = 4
    val outputChannel = 4

    val fp32Tensor = Tensor(outputChannel, inputChannel).rand

    val tensor1 = QuantizedTensor[Float](fp32Tensor,
      LinearWeightParams(outputChannel, inputChannel))
    val tensor2 = QuantizedTensor[Float](fp32Tensor,
      LinearWeightParams(outputChannel, inputChannel))

    tensor2.release()
    tensor2.set(tensor1)

    tensor2.getNativeStorage should be (tensor1.getNativeStorage)
    tensor2.getStorage should be (tensor1.getStorage)

    tensor1.release()
  }

  "A QuantizeTensor set to itself" should "work correctly" in {
    val inputChannel = 4
    val outputChannel = 4

    val fp32Tensor = Tensor(outputChannel, inputChannel).rand()
    val tensor = QuantizedTensor[Float](fp32Tensor, LinearWeightParams(outputChannel, inputChannel))

    tensor.set(tensor)

    tensor.getNativeStorage should not be 0L
    tensor.getStorage should not be null

    tensor.release()
    tensor.getNativeStorage should be (0L)
  }

  "A QuantizeTensor set" should "work correctly" in {
    val inputChannel = 4
    val outputChannel = 4

    val fp32Tensor = Tensor(outputChannel, inputChannel).rand()

    val tensor1 = QuantizedTensor[Float](fp32Tensor,
      LinearWeightParams(outputChannel, inputChannel))
    val tensor2 = QuantizedTensor[Float](fp32Tensor,
      LinearWeightParams(outputChannel, inputChannel))
    val tensor3 = QuantizedTensor[Float](fp32Tensor,
      LinearWeightParams(outputChannel, inputChannel))

    tensor2.release()
    tensor2.set(tensor1)

    tensor2.getNativeStorage should be (tensor1.getNativeStorage)
    tensor2.getStorage should be (tensor1.getStorage)

    tensor2.release()
    tensor2.set(tensor3)

    tensor2.getNativeStorage should not be tensor1.getNativeStorage

    tensor3.release()
  }

  "A QuantizeTensor serialzation" should "work correctly" in {
    val inputChannel = 4
    val outputChannel = 4

    val fp32Tensor = Tensor(outputChannel, inputChannel).rand()

    val tensor = QuantizedTensor[Float](fp32Tensor, LinearWeightParams(outputChannel, inputChannel))

    val test = SerializationUtils.clone(fp32Tensor)

    val clone = SerializationUtils.clone(tensor).asInstanceOf[QuantizedTensor[Float]]

    tensor.getStorage should be (clone.getStorage)
    tensor.maxOfRow should be (clone.maxOfRow)
    tensor.minOfRow should be (clone.minOfRow)
    tensor.sumOfRow should be (clone.sumOfRow)

    tensor.getNativeStorage should not be clone.getNativeStorage
  }
}
