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
package com.intel.analytics.bigdl.nn.mkldnn

import com.intel.analytics.bigdl.mkl.{AlgKind, Memory}
import com.intel.analytics.bigdl.nn.{SpatialAveragePooling, SpatialMaxPooling}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.BigDLSpecHelper
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG

import scala.util.Random

class MaxPoolingSpec extends BigDLSpecHelper {
  "Max Pooling test1" should "be correct" in {
    val batchSize = 2
    val input = Tensor[Float](batchSize, 480, 28, 28).apply1(e => Random.nextFloat())

    RNG.setSeed(100)
    val pool = MaxPooling(3, 3, 2, 2)
    RNG.setSeed(100)
    val layer = SpatialMaxPooling[Float](3, 3, 2, 2).ceil()

    val output2 = layer.forward(input).toTensor[Float]

    val seq = Sequential()
    seq.add(ReorderMemory(HeapData(Array(batchSize, 480, 28, 28), Memory.Format.nchw),
      HeapData(Array(batchSize, 480, 28, 28), Memory.Format.nchw)))
    seq.add(pool)
    seq.add(ReorderMemory(HeapData(Array(batchSize, 480, 14, 14), Memory.Format.nchw),
      HeapData(Array(batchSize, 480, 14, 14), Memory.Format.nchw)))
    seq.compile(Phase.TrainingPhase, Array(HeapData(Array(batchSize, 480, 28, 28),
      Memory.Format.nchw)))
    val output1 = seq.forward(input)
    output1 should be(output2)

    val grad2 = layer.backward(input, output2).toTensor[Float]
    val grad1 = seq.backward(input, output2)
    grad1 should be(grad2)
  }

  "Max Pooling test2" should "be correct" in {
    val batchSize = 2
    val input = Tensor[Float](batchSize, 64, 112, 112).apply1(e => Random.nextFloat())

    RNG.setSeed(100)
    val pool = MaxPooling(3, 3, 2, 2)
    RNG.setSeed(100)
    val layer = SpatialMaxPooling[Float](3, 3, 2, 2).ceil()

    val output2 = layer.forward(input).toTensor[Float]

    val seq = Sequential()
    seq.add(ReorderMemory(HeapData(Array(batchSize, 64, 112, 112), Memory.Format.nchw),
      HeapData(Array(batchSize, 64, 112, 112), Memory.Format.nchw)))
    seq.add(pool)
    seq.add(ReorderMemory(HeapData(Array(batchSize, 64, 56, 56), Memory.Format.nchw),
      HeapData(Array(batchSize, 64, 56, 56), Memory.Format.nchw)))
    seq.compile(Phase.TrainingPhase, Array(HeapData(Array(batchSize, 64, 112, 112),
      Memory.Format.nchw)))
    val output1 = seq.forward(input)
    output1 should be(output2)

    val grad2 = layer.backward(input, output2).toTensor[Float]
    val grad1 = seq.backward(input, output2)
    grad1 should be(grad2)
  }
}
