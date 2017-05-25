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
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.{RandomGenerator, T, Table}

import scala.util.Random

class GraphNodeSpec extends FlatSpec with Matchers {

  "Inception bn+GraphNode" should "generate correct output" in {
    Random.setSeed(3)
    val input = Tensor[Float](4, 3, 224, 224).apply1(e => Random.nextFloat())
    val labels = Tensor[Float](4).apply1(e => Random.nextInt(1000))

    val seed = 100
    RNG.setSeed(seed)
    val model = Inception.getModel[Float](1000, "inception-bn")
    RNG.setSeed(seed)
    val model2 = Inception.getModel[Float](1000, "inception-bn")
    val start = System.nanoTime()
    val graphModel = model2.toGraph()
    val end = System.nanoTime()
    val scalaTime = end - start
    println("Inception bn Module to Graph takes time " + scalaTime/1e9 + "s")

    val outputStart1 = System.nanoTime()
    val output1 = model.forward(input).toTensor[Float]
    val outputEnd1 = System.nanoTime()
    println("Forward of Original Module takes time " + (outputEnd1-outputStart1)/1e9 + "s")
    val outputStart2 = System.nanoTime()
    val output2 = graphModel.forward(input).toTensor[Float]
    val outputEnd2 = System.nanoTime()
    println("Forward of Graph Module takes time " + (outputEnd2-outputStart2)/1e9 + "s")
    output1 should be (output2)

    val criterion = new ClassNLLCriterion[Float]()
    val loss = criterion.forward(output1, labels)
    val gradOutput = criterion.backward(output1, labels)

    val gradOutputStart1 = System.nanoTime()
    val gradInput1 = model.backward(input, gradOutput)
    val gradOutputEnd1 = System.nanoTime()
    println("Backward of Original Module takes time " + (gradOutputEnd1-gradOutputStart1)/1e9 + "s")

    val gradOutputStart2 = System.nanoTime()
    val gradInput2 = graphModel.backward(input, gradOutput)
    val gradOutputEnd2 = System.nanoTime()
    println("Backward of Graph Module takes time " + (gradOutputEnd2-gradOutputStart2)/1e9 + "s")

    gradInput1 should be (gradInput2)
  }

  "ResNet+GraphNode" should "generate correct output" in {
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
      "depth" -> depth, "dataset" -> DatasetType.ImageNet))
    RNG.setSeed(seed)
    val model2 = ResNet(classNum, T("shortcutType" -> ShortcutType.B,
      "depth" -> depth, "dataset" -> DatasetType.ImageNet))

    val (weights, grad) = model.getParameters()
    val (w, g) = model2.getParameters()
    w.copy(weights)
    val start = System.nanoTime()
    val graphModel = model2.toGraph()
    val end = System.nanoTime()
    val scalaTime = end - start
    println("Module to Graph takes time " + scalaTime/1e9 + "s")

    val outputStart1 = System.nanoTime()
    val output1 = model.forward(input).toTensor[Float]
    val outputEnd1 = System.nanoTime()
    println("Forward of Original Module takes time " + (outputEnd1-outputStart1)/1e9 + "s")
    val outputStart2 = System.nanoTime()
    val output2 = graphModel.forward(input).toTensor[Float]
    val outputEnd2 = System.nanoTime()
    println("Forward of Graph Module takes time " + (outputEnd2-outputStart2)/1e9 + "s")
    output1 should be (output2)

    val criterion = new ClassNLLCriterion[Float]()
    val loss = criterion.forward(output1, labels)
    val gradOutput = criterion.backward(output1, labels)

    val gradOutputStart1 = System.nanoTime()
    val gradInput1 = model.backward(input, gradOutput)
    val gradOutputEnd1 = System.nanoTime()
    println("Backward of Original Module takes time " + (gradOutputEnd1-gradOutputStart1)/1e9 + "s")

    val gradOutputStart2 = System.nanoTime()
    val gradInput2 = graphModel.backward(input, gradOutput)
    val gradOutputEnd2 = System.nanoTime()
    println("Backward of Graph Module takes time " + (gradOutputEnd2-gradOutputStart2)/1e9 + "s")

    gradInput1 should be (gradInput2)
  }

  "AlexNet+GraphNode" should "generate correct output" in {
    Random.setSeed(1)
    val input = Tensor[Float](8, 3, 224, 224).apply1(e => Random.nextFloat())
    val labels = Tensor[Float](8).apply1(e => Random.nextInt(100))

    val seed = 100
    RNG.setSeed(seed)
    val model = AlexNet_OWT(1000, false, true)
    RNG.setSeed(seed)
    val model2 = AlexNet_OWT(1000, false, true)
    val start = System.nanoTime()
    val graphModel = model2.toGraph()
    val end = System.nanoTime()
    val scalaTime = end - start
    println("Module to Graph takes time " + scalaTime/1e9 + "s")

    val outputStart1 = System.nanoTime()
    val output1 = model.forward(input).toTensor[Float]
    val outputEnd1 = System.nanoTime()
    println("Forward of Original Module takes time " + (outputEnd1-outputStart1)/1e9 + "s")
    val outputStart2 = System.nanoTime()
    val output2 = graphModel.forward(input).toTensor[Float]
    val outputEnd2 = System.nanoTime()
    println("Forward of Graph Module takes time " + (outputEnd2-outputStart2)/1e9 + "s")
    output1 should be (output2)

    val criterion = new ClassNLLCriterion[Float]()
    val loss = criterion.forward(output1, labels)
    val gradOutput = criterion.backward(output1, labels)

    val gradOutputStart1 = System.nanoTime()
    val gradInput1 = model.backward(input, gradOutput)
    val gradOutputEnd1 = System.nanoTime()
    println("Backward of Original Module takes time " + (gradOutputEnd1-gradOutputStart1)/1e9 + "s")

    val gradOutputStart2 = System.nanoTime()
    val gradInput2 = graphModel.backward(input, gradOutput)
    val gradOutputEnd2 = System.nanoTime()
    println("Backward of Graph Module takes time " + (gradOutputEnd2-gradOutputStart2)/1e9 + "s")

    gradInput1 should be (gradInput2)
  }
}
