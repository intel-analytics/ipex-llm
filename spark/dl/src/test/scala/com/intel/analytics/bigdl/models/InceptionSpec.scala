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

package com.intel.analytics.bigdl.models

import com.intel.analytics.bigdl.models.inception._
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Graph, Input}
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.optim.SGD
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.torch.{TH, TorchSpec}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.bigdl.numeric.NumericFloat

import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.math._
import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class InceptionSpec extends FlatSpec with BeforeAndAfter with Matchers {
  "Inception ModelCaffe" should "init right" in {
    RNG.setSeed(1024)

    val random = new Random
    random.setSeed(1024)

    val input = Tensor[Float](4, 3, 224, 224).apply1(e => random.nextFloat())
    val labels = Tensor[Float](4).apply1(e => random.nextInt(1000))

    val model = Inception.getModelCaffe[Float](1000)

    val criterion = new ClassNLLCriterion[Float]()

    model.zeroGradParameters()
    val output = model.forward(input).toTensor[Float]
    val loss = criterion.forward(output, labels)

    // since we already set the seed, the loss should match exactly
    loss should be (6.8930426f)
  }

  "InceptionV1 " should "init right" in {
    RNG.setSeed(1024)

    val random = new Random
    random.setSeed(1024)

    val input = Tensor[Float](4, 3, 224, 224).apply1(e => random.nextFloat())
    val labels = Tensor[Float](4).apply1(e => random.nextInt(1000))

    val model = Inception_v1(1000)

    val criterion = new ClassNLLCriterion[Float]()

    model.zeroGradParameters()
    val output = model.forward(input).toTensor[Float]
    val loss = criterion.forward(output, labels)

    // since we already set the seed, the loss should match exactly
    loss should be (6.9011583f)
  }

  "Inception_Layer_V1 graph" should "be correct" in {
    val batchSize = 8
    RNG.setSeed(1000)
    val model = Inception_Layer_v1(2, T(T(4), T(96, 128), T(16, 32), T(32)), "conv")
    RNG.setSeed(1000)
    val input = Input()
    val f1 = Inception_Layer_v1(input, 2, T(T(4), T(96, 128), T(16, 32), T(32)), "conv")
    val graphModel = Graph(input, f1)

    val inputData = Tensor(batchSize, 2, 4, 4).rand()
    val gradOutput = Tensor(batchSize, 256, 4, 4).rand()

    val output1 = model.forward(inputData).toTensor[Float]
    val output2 = graphModel.forward(inputData).toTensor[Float]
    output1 should be(output2)

    val gradInput1 = model.backward(inputData, gradOutput).toTensor[Float]
    val gradInput2 = graphModel.backward(inputData, gradOutput).toTensor[Float]
    gradInput1 should be(gradInput2)

    model.getParametersTable()[Table]("conv1x1")[Tensor[Float]]("gradWeight") should be(
      graphModel.getParametersTable()[Table]("conv1x1")[Tensor[Float]]("gradWeight")
    )

    model.getParametersTable()[Table]("conv3x3_reduce")[Tensor[Float]]("gradWeight") should be(
      graphModel.getParametersTable()[Table]("conv3x3_reduce")[Tensor[Float]]("gradWeight")
    )

    model.getParametersTable()[Table]("conv3x3")[Tensor[Float]]("gradWeight") should be(
      graphModel.getParametersTable()[Table]("conv3x3")[Tensor[Float]]("gradWeight")
    )

    model.getParametersTable()[Table]("conv5x5_reduce")[Tensor[Float]]("gradWeight") should be(
      graphModel.getParametersTable()[Table]("conv5x5_reduce")[Tensor[Float]]("gradWeight")
    )

    model.getParametersTable()[Table]("conv5x5")[Tensor[Float]]("gradWeight") should be(
      graphModel.getParametersTable()[Table]("conv5x5")[Tensor[Float]]("gradWeight")
    )

    model.getParametersTable()[Table]("convpool_proj")[Tensor[Float]]("gradWeight") should be(
      graphModel.getParametersTable()[Table]("convpool_proj")[Tensor[Float]]("gradWeight")
    )
  }

  "Inception graph" should "be correct" in {
    val batchSize = 2
    RNG.setSeed(1000)
    val model = Inception_v1_NoAuxClassifier(1000, false)
    RNG.setSeed(1000)
    val graphModel = Inception_v1_NoAuxClassifier.graph(1000, false)

    val input = Tensor[Float](batchSize, 3, 224, 224).apply1(e => Random.nextFloat())
    val gradOutput = Tensor[Float](batchSize, 1000).apply1(e => Random.nextFloat())

    val output1 = model.forward(input).toTensor[Float]
    val output2 = graphModel.forward(input).toTensor[Float]
    output1 should be(output2)

    val gradInput1 = model.backward(input, gradOutput).toTensor[Float]
    val gradInput2 = graphModel.backward(input, gradOutput).toTensor[Float]
    gradInput1 should be(gradInput2)

    model.getParametersTable().equals(graphModel.getParametersTable()) should be (true)
  }

  "Inception_v1 graph" should "be correct" in {
    val batchSize = 1
    RNG.setSeed(1000)
    val model = Inception_v1(1000, false)
    RNG.setSeed(1000)
    val graphModel = Inception_v1.graph(1000, false)

    val input = Tensor[Float](batchSize, 3, 224, 224).apply1(e => Random.nextFloat())
    val gradOutput = Tensor[Float](batchSize, 3000).apply1(e => Random.nextFloat())

    val output1 = model.forward(input).toTensor[Float]
    val output2 = graphModel.forward(input).toTensor[Float]
    output1 should be(output2)

    val gradInput1 = model.backward(input, gradOutput)
    val gradInput2 = graphModel.backward(input, gradOutput)
    gradInput1 should be(gradInput2)
  }

  "Inception_Layer_V2 graph" should "be correct" in {
    val batchSize = 8
    RNG.setSeed(1000)
    val model = Inception_Layer_v2(2, T(T(4), T(96, 128), T(16, 32), T("avg", 32)), "conv")
    RNG.setSeed(1000)
    val input1 = Input()
    val f1 = Inception_Layer_v2(input1, 2, T(T(4), T(96, 128), T(16, 32), T("avg", 32)), "conv")
    val graphModel = Graph(input1, f1)

    val input = Tensor(batchSize, 2, 4, 4).rand()
    val gradOutput = Tensor(batchSize, 256, 4, 4).rand()

    val output1 = model.forward(input).toTensor[Float]
    val output2 = graphModel.forward(input).toTensor[Float]
    output1 should be(output2)

    val gradInput1 = model.backward(input, gradOutput).toTensor[Float]
    val gradInput2 = graphModel.backward(input, gradOutput).toTensor[Float]
    gradInput1 should be(gradInput2)

    model.getParametersTable().equals(graphModel.getParametersTable()) should be (true)
  }

  "Inception_v2_NoAuxClassifier graph" should "be correct" in {
    val batchSize = 2
    RNG.setSeed(1000)
    val model = Inception_v2_NoAuxClassifier(1000)
    RNG.setSeed(1000)
    val graphModel = Inception_v2_NoAuxClassifier.graph(1000)

    val input = Tensor[Float](batchSize, 3, 224, 224).apply1(e => Random.nextFloat())
    val gradOutput = Tensor[Float](batchSize, 1000).apply1(e => Random.nextFloat())

    val output1 = model.forward(input).toTensor[Float]
    val output2 = graphModel.forward(input).toTensor[Float]
    output1 should be(output2)

    val gradInput1 = model.backward(input, gradOutput).toTensor[Float]
    val gradInput2 = graphModel.backward(input, gradOutput).toTensor[Float]
    gradInput1 should be(gradInput2)

    model.getParametersTable().equals(graphModel.getParametersTable()) should be (true)
  }

  "Inception_v2 graph" should "be correct" in {
    val batchSize = 2
    RNG.setSeed(1000)
    val model = Inception_v2(1000)
    RNG.setSeed(1000)
    val graphModel = Inception_v2.graph(1000)

    val input = Tensor[Float](batchSize, 3, 224, 224).apply1(e => Random.nextFloat())
    val gradOutput = Tensor[Float](batchSize, 3000).apply1(e => Random.nextFloat())

    val output1 = model.forward(input).toTensor[Float]
    val output2 = graphModel.forward(input).toTensor[Float]
    output1 should be(output2)

    val gradInput1 = model.updateGradInput(input, gradOutput)
    val gradInput2 = graphModel.updateGradInput(input, gradOutput)
    gradInput1 should be(gradInput2)
  }
}
