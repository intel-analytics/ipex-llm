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

import com.intel.analytics.bigdl.quantization.Quantization
import org.scalatest.{FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Parallel
class QuantizeTensorSpec extends FlatSpec with Matchers {
  "A QuantizeTensor set to empty" should "work correctly" in {
    val inputChannel = 4
    val outputChannel = 4
    val batchSize = 3

    val threshold = 64.0f

    val tensor = QuantizeTensor[Float](outputChannel, inputChannel)
    val byteArray = Array.fill[Byte](inputChannel * outputChannel)(0)

    tensor.setStorageInJni(Quantization.FCKernelDescInit(batchSize, outputChannel))
    tensor.setStorage(byteArray)

    val minArray = Array.fill[Float](outputChannel)(0)
    val maxArray = Array.fill[Float](outputChannel)(0)

    Quantization.FCKernelLoadFromModel(tensor.getStorageInJni, byteArray,
      minArray, maxArray, outputChannel, inputChannel, threshold, Quantization.NCHW)

    tensor.set()

    tensor.getStorageInJni should be (0L)
    tensor.getStorage should be (null)

    tensor.release()
  }

  "A QuantizeTensor set to other tensor" should "work correctly" in {
    val inputChannel = 4
    val outputChannel = 4
    val batchSize = 3

    val threshold = 64.0f

    val tensor1 = QuantizeTensor[Float](outputChannel, inputChannel)
    val tensor2 = QuantizeTensor[Float](outputChannel, inputChannel)

    val byteArray = Array.fill[Byte](inputChannel * outputChannel)(0)

    tensor1.setStorageInJni(Quantization.FCKernelDescInit(batchSize, outputChannel))
    tensor2.setStorageInJni(Quantization.FCKernelDescInit(batchSize, outputChannel))
    tensor1.setStorage(byteArray)
    tensor2.setStorage(byteArray)

    val minArray = Array.fill[Float](outputChannel)(0)
    val maxArray = Array.fill[Float](outputChannel)(0)

    Quantization.FCKernelLoadFromModel(tensor1.getStorageInJni, byteArray,
      minArray, maxArray, outputChannel, inputChannel, threshold, Quantization.NCHW)
    Quantization.FCKernelLoadFromModel(tensor2.getStorageInJni, byteArray,
      minArray, maxArray, outputChannel, inputChannel, threshold, Quantization.NCHW)

    tensor2.set(tensor1)

    tensor2.getStorageInJni should be (tensor1.getStorageInJni)
    tensor2.getStorage should be (tensor1.getStorage)

    tensor1.release()
    tensor2.release()
  }

  "A QuantizeTensor set to itself" should "work correctly" in {
    val inputChannel = 4
    val outputChannel = 4
    val batchSize = 3

    val threshold = 64.0f

    val tensor = QuantizeTensor[Float](outputChannel, inputChannel)
    val byteArray = Array.fill[Byte](inputChannel * outputChannel)(0)

    tensor.setStorageInJni(Quantization.FCKernelDescInit(batchSize, outputChannel))
    tensor.setStorage(byteArray)

    val minArray = Array.fill[Float](outputChannel)(0)
    val maxArray = Array.fill[Float](outputChannel)(0)

    Quantization.FCKernelLoadFromModel(tensor.getStorageInJni, byteArray,
      minArray, maxArray, outputChannel, inputChannel, threshold, Quantization.NCHW)

    tensor.set(tensor)

    tensor.getStorageInJni should not be 0L
    tensor.getStorage should not be null

    tensor.release()

    tensor.getStorageInJni should be (0L)
  }

  "A QuantizeTensor set" should "work correctly" in {
    val inputChannel = 4
    val outputChannel = 4
    val batchSize = 3

    val threshold = 64.0f

    val tensor1 = QuantizeTensor[Float](outputChannel, inputChannel)
    val tensor2 = QuantizeTensor[Float](outputChannel, inputChannel)

    val byteArray = Array.fill[Byte](inputChannel * outputChannel)(0)

    tensor1.setStorageInJni(Quantization.FCKernelDescInit(batchSize, outputChannel))
    tensor2.setStorageInJni(Quantization.FCKernelDescInit(batchSize, outputChannel))
    tensor1.setStorage(byteArray)
    tensor2.setStorage(byteArray)

    val minArray = Array.fill[Float](outputChannel)(0)
    val maxArray = Array.fill[Float](outputChannel)(0)

    Quantization.FCKernelLoadFromModel(tensor1.getStorageInJni, byteArray,
      minArray, maxArray, outputChannel, inputChannel, threshold, Quantization.NCHW)
    Quantization.FCKernelLoadFromModel(tensor2.getStorageInJni, byteArray,
      minArray, maxArray, outputChannel, inputChannel, threshold, Quantization.NCHW)

    tensor2.set(tensor1)

    tensor2.getStorageInJni should be (tensor1.getStorageInJni)
    tensor2.getStorage should be (tensor1.getStorage)

    tensor2.setStorageInJni(Quantization.FCKernelDescInit(batchSize, outputChannel))

    tensor2.getStorageInJni should not be tensor1.getStorageInJni

    tensor1.release()
    tensor2.release()
  }
}
