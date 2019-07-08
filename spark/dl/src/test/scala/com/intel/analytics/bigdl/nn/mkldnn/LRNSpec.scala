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

import com.intel.analytics.bigdl.mkl.{DataType, Memory}
import com.intel.analytics.bigdl.nn.SpatialCrossMapLRN
import com.intel.analytics.bigdl.nn.mkldnn.Phase.{InferencePhase, TrainingPhase}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.BigDLSpecHelper
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import org.apache.commons.lang3.SerializationUtils

import scala.util.Random

class LRNSpec extends BigDLSpecHelper {
  "LRNDnn with format=nchw" should "work correctly" in {
    val batchSize = 2
    val input = Tensor[Float](batchSize, 7, 3, 3).apply1(e => Random.nextFloat())
    val gradOutput = Tensor[Float](batchSize, 7, 3, 3).apply1(e => Random.nextFloat())

    RNG.setSeed(100)
    val lrnDnn = LRN(5, 0.0001, 0.75, 1.0)
    RNG.setSeed(100)
    val lrnBLAS = SpatialCrossMapLRN[Float](5, 0.0001, 0.75, 1.0)

    val output2 = lrnBLAS.forward(input)
    val grad2 = lrnBLAS.updateGradInput(input, gradOutput)

    val seq = Sequential()
    seq.add(ReorderMemory(HeapData(Array(batchSize, 7, 3, 3), Memory.Format.nchw),
      HeapData(Array(batchSize, 7, 3, 3), Memory.Format.nchw)))
    seq.add(lrnDnn)
    seq.add(ReorderMemory(HeapData(Array(batchSize, 7, 3, 3), Memory.Format.nchw),
      HeapData(Array(batchSize, 7, 3, 3), Memory.Format.nchw)))
    seq.compile(Phase.TrainingPhase, Array(HeapData(Array(batchSize, 7, 3, 3), Memory.Format.nchw)))
    val output = seq.forward(input)
    output.asInstanceOf[Tensor[Float]] should be(output2)
    val grad1 = seq.backward(input, gradOutput)
    grad1.asInstanceOf[Tensor[Float]] should be(grad2)
  }

  "lrn with java serialization" should "work correctly" in {
    val batchSize = 2
    val inputShape = Array(batchSize, 7, 3, 3)
    val input = Tensor[Float](batchSize, 7, 3, 3).rand(-1, 1)
    val gradOutput = Tensor[Float](batchSize, 7, 3, 3).rand(-1, 1)

    val lrn = LRN(5, 0.0001, 0.75, 1.0)
    lrn.setRuntime(new MklDnnRuntime)
    lrn.initFwdPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)
    lrn.initBwdPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)
    lrn.initGradWPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)

    lrn.forward(input)

    val cloned = SerializationUtils.clone(lrn)
    cloned.setRuntime(new MklDnnRuntime)
    cloned.initFwdPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)
    cloned.initBwdPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)
    cloned.initGradWPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)

    cloned.forward(input)

    Tools.dense(lrn.output) should be (Tools.dense(cloned.output))
  }

  "LRN in int8 model" should "work correctly" in {
    RNG.setSeed(100)

    val inputShape = Array(4, 8, 3, 3)
    val input = Tensor[Float](inputShape).rand(-1, 1)

    val int8NativeData = NativeData(inputShape, Memory.Format.nhwc, DataType.S8)
    int8NativeData.setMask(0)
    int8NativeData.setScales(Array(127.0f / input.clone().abs().max()))
    val reorderToInt8 = ReorderMemory(int8NativeData)

    val seqInt8 = Sequential()
      .add(Input(inputShape, Memory.Format.nchw))
      .add(reorderToInt8)
      .add(LRN(8, 0.0001, 0.75, 1.0))
      .add(ReorderMemory(HeapData(inputShape, Memory.Format.nchw)))

    seqInt8.compile(InferencePhase)

    seqInt8.forward(input)

    val seqFP32 = Sequential()
      .add(Input(inputShape, Memory.Format.nchw))
      .add(LRN(8, 0.0001, 0.75, 1.0))
      .add(ReorderMemory(HeapData(inputShape, Memory.Format.nchw)))

    seqFP32.compile(InferencePhase)
    seqFP32.forward(input)

    // here, the 1e-2 is experience value
    Equivalent.nearequals(seqInt8.output.toTensor, seqFP32.output.toTensor, 1e-2) should be (true)

    seqInt8.release()
    seqFP32.release()

    println()
  }
}
