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

package com.intel.analytics.bigdl.utils


import com.intel.analytics.bigdl.example.loadmodel.AlexNet_OWT
import com.intel.analytics.bigdl.models.Inception
import com.intel.analytics.bigdl.models.resnet.ResNet
import com.intel.analytics.bigdl.models.resnet.ResNet.{DatasetType, ShortcutType}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor

import scala.util.Random

class GraphNodeSpec extends FlatSpec with Matchers {

  "Inception bn to Graph" should "generate correct output" in {
    val batchSize = 2
    Random.setSeed(3)
    val input = Tensor[Float](batchSize, 3, 224, 224).apply1(e => Random.nextFloat())
    val labels = Tensor[Float](batchSize).apply1(e => Random.nextInt(1000))

    val inputNew = input.clone()

    val seed = 100
    RNG.setSeed(seed)
    val model = Inception.getModel[Float](1000, "inception-bn")
    RNG.setSeed(seed)
    val model2 = Inception.getModel[Float](1000, "inception-bn")
    val graphModel = model2.toGraph()

    val output1 = model.forward(input).toTensor[Float]
    val output2 = graphModel.forward(input).toTensor[Float]
    output1 should be(output2)

    val criterion = new ClassNLLCriterion[Float]()
    val loss = criterion.forward(output1, labels)
    val gradOutput = criterion.backward(output1, labels)


    val gradInput1 = model.backward(input, gradOutput).toTensor
    val gradInput2 = graphModel.backward(input, gradOutput).toTensor

    val arr1 = gradInput1.storage().array()
    val arr2 = gradInput2.storage().array()

    for (i <- 0 to (arr1.length-1)) {
      arr1(i) should be(arr2(i) +- 1e-5f)
    }

    // gradInput1.equals(gradInput2) should be(true)
  }

  "ResNet to Graph" should "generate correct output" in {
    val inputSeed = 1
    val depth = 18
    val batchSize = 4
    val modelSeed = 101
    Random.setSeed(inputSeed)
    val classNum: Int = 1000
    val input = Tensor[Float](batchSize, 3, 224, 224).apply1( e => Random.nextFloat())
    val labels = Tensor[Float](batchSize).apply1(e => Random.nextInt(classNum))
    val seed = modelSeed
    RNG.setSeed(seed)
    val model = ResNet(classNum, T("shortcutType" -> ShortcutType.B,
      "depth" -> depth, "dataSet" -> DatasetType.ImageNet))
    RNG.setSeed(seed)
    val model2 = ResNet(classNum, T("shortcutType" -> ShortcutType.B,
      "depth" -> depth, "dataSet" -> DatasetType.ImageNet))

    val (weights, grad) = model.getParameters()
    val (w, g) = model2.getParameters()
    w.copy(weights)
    val graphModel = model2.toGraph()

    val output1 = model.forward(input).toTensor[Float]
    val output2 = graphModel.forward(input).toTensor[Float]
    output1 should be (output2)

    val criterion = new ClassNLLCriterion[Float]()
    val loss = criterion.forward(output1, labels)
    val gradOutput = criterion.backward(output1, labels)

    val gradInput1 = model.backward(input, gradOutput)
    val gradInput2 = graphModel.backward(input, gradOutput)

    gradInput1 should be (gradInput2)
  }

  "AlexNet to Graph" should "generate correct output" in {
    Random.setSeed(1)
    val input = Tensor[Float](8, 3, 224, 224).apply1(e => Random.nextFloat())
    val labels = Tensor[Float](8).apply1(e => Random.nextInt(100))

    val seed = 100
    RNG.setSeed(seed)
    val model = AlexNet_OWT(1000, false, true)
    RNG.setSeed(seed)
    val model2 = AlexNet_OWT(1000, false, true)
    val graphModel = model2.toGraph()

    val output1 = model.forward(input).toTensor[Float]
    val output2 = graphModel.forward(input).toTensor[Float]
    output1 should be (output2)

    val criterion = new ClassNLLCriterion[Float]()
    val loss = criterion.forward(output1, labels)
    val gradOutput = criterion.backward(output1, labels)

    val gradInput1 = model.backward(input, gradOutput)
    val gradInput2 = graphModel.backward(input, gradOutput)

    gradInput1 should be (gradInput2)
  }

  "Recurrent+LSTM to graph" should "generate correct output" in {
    Random.setSeed(1)
    val input = Tensor[Float](8, 128, 128).apply1(e => Random.nextFloat())
    val gradOutput = Tensor[Float](8, 128, 128).apply1(e => Random.nextFloat())

    val seed = 100
    val inputSize = 128
    val hiddenSize = 128
    val outputSize = 128

    RNG.setSeed(seed)
    val model = Sequential[Float]()
    model.add(Recurrent[Float]()
      .add(RnnCell[Float](inputSize, hiddenSize, Tanh[Float]())))
      .add(TimeDistributed[Float](Linear[Float](hiddenSize, outputSize)))

    val model2 = model.cloneModule()
    val graphModel = model2.toGraph()

    val output1 = model.forward(input).toTensor[Float]
    val output2 = graphModel.forward(input).toTensor[Float]
    output1 should be (output2)

    val gradInput1 = model.backward(input, gradOutput)
    val gradInput2 = graphModel.backward(input, gradOutput)

    gradInput1 should be (gradInput2)
  }

  "ParallelTable to graph" should "generate correct output" in {
    Random.setSeed(1)
    val batchSize = 4
    val hiddenSize = 12
    val input = T(Tensor[Float](batchSize, hiddenSize).apply1(e => Random.nextFloat()),
      Tensor[Float](batchSize, hiddenSize).apply1(e => Random.nextFloat()))
    val gradOutput = T(Tensor[Float](batchSize, hiddenSize).apply1(e => Random.nextFloat()),
      Tensor[Float](batchSize, hiddenSize).apply1(e => Random.nextFloat()))

    val seed = 100
    RNG.setSeed(seed)
    val model = ParallelTable()
        .add(Linear(hiddenSize, hiddenSize))
        .add(Linear(hiddenSize, hiddenSize))

    val model2 = model.cloneModule()
    val graphModel = model2.toGraph(Input(), Input())

    val output1 = model.forward(input)
    val output2 = graphModel.forward(input)
    output1 should be (output2)

    val gradInput1 = model.backward(input, gradOutput)
    val gradInput2 = graphModel.backward(input, gradOutput)

    gradInput1 should be (gradInput2)
  }

  "CAddTable to graph" should "generate correct output" in {
    val module = CAddTable[Float]().toGraph(Input(), Input())
    val scalar = Tensor[Float](Array(2.0f), Array[Int]())
    val tensor = Tensor[Float](T(1, 2, 3))
    val output = module.forward(T(scalar, tensor))
    output should be(Tensor[Float](T(3, 4, 5)))
    val grads = module.backward(T(scalar, tensor), Tensor[Float](T(1, 2, 3))).toTable
    grads[Tensor[Float]](1).value() should be(6)
    grads[Tensor[Float]](2) should be(Tensor[Float](T(1, 2, 3)))
  }

  "Bottle to graph" should "generate correct output" in {
    val seed = 100
    RNG.setSeed(seed)
    val model = Bottle[Double](Linear[Double](10, 2), 2, 2)
    model.add(Linear(10, 2))

    val input = Tensor[Double](4, 5, 10).apply1(_ => Random.nextDouble())
    val gradOutput = Tensor[Double](4, 10).apply1(_ => Random.nextDouble())

    val graphModel = model.cloneModule().toGraph(Input())

    val output1 = model.forward(input)
    val output2 = graphModel.forward(input)
    output1 should be (output2)

    val gradInput1 = model.backward(input, gradOutput)
    val gradInput2 = graphModel.backward(input, gradOutput)

    gradInput1 should be (gradInput2)
  }

}
