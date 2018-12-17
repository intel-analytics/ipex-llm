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

import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.nn.mkldnn.Phase.InferencePhase
import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor

class FusionSpec extends FlatSpec with Matchers {
  "Conv with relu" should "work correctly" in {
    val batchSize = 2
    val input = Tensor[Float](batchSize, 3, 224, 224).fill(1.0f)

    val inputShape = Array(batchSize, 3, 224, 224)
    val outputShape = Array(batchSize, 64, 112, 112)

    val conv1 = SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3, 1, false)
    val reorder1 = ReorderMemory(HeapData(inputShape, Memory.Format.nchw))
    val reorder11 = ReorderMemory(HeapData(outputShape, Memory.Format.nchw))
    val model1 = Sequential().add(reorder1).add(conv1).add(ReLU()).add(reorder11)
    model1.compile(InferencePhase, Array(HeapData(inputShape, Memory.Format.nchw)))

    System.setProperty("bigdl.mkldnn.fusion.convrelu", "true")
    val conv2 = SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3, 1, false,
      initWeight = conv1.weight.native, initBias = conv1.bias.native)
    val reorder2 = ReorderMemory(HeapData(inputShape, Memory.Format.nchw))
    val reorder22 = ReorderMemory(HeapData(outputShape, Memory.Format.nchw))
    val model2 = Sequential().add(reorder2).add(conv2).add(ReLU()).add(reorder22)
    model2.compile(InferencePhase, Array(HeapData(inputShape, Memory.Format.nchw)))
    System.setProperty("bigdl.mkldnn.fusion.convrelu", "false")

    model1.evaluate()
    model2.evaluate()

    model1.forward(input)
    model2.forward(input)

    model1.output should be (model2.output)
    model1.modules.length should be (model2.modules.length + 1)
  }

  "Conv Bn merge" should "work correctly" in {
    val batchSize = 4
    val inputShape = Array(batchSize, 3, 224, 224)
    val outputShape = Array(batchSize, 64, 112, 112)
    val input = Tensor[Float](batchSize, 3, 224, 224).fill(1.0f)

    val runningMean = Tensor[Float](64).rand(-1, 1)
    val runningVar = Tensor[Float](64).fill(100)
    val initWeight = Tensor[Float]().resize(Array(64, 3, 7, 7)).rand(-1, 1)
    val initBias = Tensor[Float]().resize(Array(64)).rand(-100, 100)

    val conv1 = SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3, 1, false, initWeight = initWeight,
      initBias = initBias)
    val bn1 = SpatialBatchNormalization(64)
    bn1.runningMean.copy(runningMean)
    bn1.runningVariance.copy(runningVar)
    bn1.scaleFactor = 1.0f
    val reorder1 = ReorderMemory(HeapData(inputShape, Memory.Format.nchw))
    val reorder11 = ReorderMemory(HeapData(outputShape, Memory.Format.nchw))
    val model1 = Sequential().add(reorder1).add(conv1).add(bn1).add(reorder11)
    model1.compile(InferencePhase, Array(HeapData(inputShape, Memory.Format.nchw)))

    System.setProperty("bigdl.mkldnn.fusion.convbn", "true")
    val conv2 = SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3, 1, false,
      initWeight = conv1.weight.native, initBias = conv1.bias.native)
    val bn2 = SpatialBatchNormalization(64)
    bn2.runningMean.copy(runningMean)
    bn2.runningVariance.copy(runningVar)
    bn2.scaleFactor = 1.0f
    val reorder2 = ReorderMemory(HeapData(inputShape, Memory.Format.nchw))
    val reorder22 = ReorderMemory(HeapData(outputShape, Memory.Format.nchw))
    val model2 = Sequential().add(reorder2).add(conv2).add(bn2).add(reorder22)
    model2.compile(InferencePhase, Array(HeapData(inputShape, Memory.Format.nchw)))
    System.setProperty("bigdl.mkldnn.fusion.convbn", "false")

    model1.evaluate()
    model2.evaluate()

    model1.forward(input)
    model2.forward(input)

    Equivalent.nearequals(model1.output.toTensor, model2.output.toTensor, 1e-5) should be (true)
    model1.modules.length should be (model2.modules.length + 1)
  }

  "Conv sum fusion" should "work correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericFloat

    val input = Tensor[Float](2, 1, 6, 6).rand(-1, 1)
    val inputShape = Array(2, 1, 6, 6)
    val outputShape = Array(2, 3, 4, 4)

    val initWeight = Tensor[Float](3, 1, 2, 2).fill(1)
    val initBias = Tensor[Float](3).fill(0)

    val conv1 = SpatialConvolution(1, 3, 2, 2, 2, 2, 1, 1, 1, initWeight = initWeight,
      initBias = initBias)
    val conv2 = SpatialConvolution(1, 3, 2, 2, 2, 2, 1, 1, 1, initWeight = initWeight,
      initBias = initBias)
    val conv3 = SpatialConvolution(1, 3, 2, 2, 2, 2, 1, 1, 1, initWeight = initWeight,
      initBias = initBias)
    val conv4 = SpatialConvolution(1, 3, 2, 2, 2, 2, 1, 1, 1, initWeight = initWeight,
      initBias = initBias)

    val reorder1 = ReorderMemory(HeapData(outputShape, Memory.Format.nchw))
    val reorder2 = ReorderMemory(HeapData(outputShape, Memory.Format.nchw))

    val model1 = Sequential()
      .add(ConcatTable()
        .add(conv1)
        .add(conv2))
      .add(CAddTable())
      .add(ReLU())
      .add(reorder1)
    model1.compile(InferencePhase, Array(HeapData(inputShape, Memory.Format.nchw)))

    System.setProperty("bigdl.mkldnn.fusion.convsum", "true")
    System.setProperty("bigdl.mkldnn.fusion.convrelu", "true")
    val model2 = Sequential()
      .add(ConcatTable()
        .add(conv3)
        .add(conv4))
      .add(CAddTable())
      .add(ReLU())
      .add(reorder2)

    model1.evaluate()
    model2.evaluate()

    model2.compile(InferencePhase, Array(HeapData(inputShape, Memory.Format.nchw)))
    System.setProperty("bigdl.mkldnn.fusion.convsum", "false")
    System.setProperty("bigdl.mkldnn.fusion.convrelu", "false")

    model1.forward(input)
    model2.forward(input)

    model1.output should be (model2.output)
  }

}
