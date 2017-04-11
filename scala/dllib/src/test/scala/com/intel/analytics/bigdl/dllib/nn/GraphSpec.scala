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
package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.models.inception.{Inception_Layer_v1, Inception_v1}
import com.intel.analytics.bigdl.models.resnet.{Convolution, ResNet}
import com.intel.analytics.bigdl.models.resnet.ResNet.{ShortcutType, iChannels}
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.{RandomGenerator, T, Table}

import scala.reflect.ClassTag

@com.intel.analytics.bigdl.tags.Parallel
class GraphSpec extends FlatSpec with Matchers {
  "Graph init" should "throw exceptions when there's cycle" in {
    val fc1 = Linear(4, 2).apply()
    val relu1 = ReLU().apply(fc1)
    relu1 -> fc1

    intercept[IllegalArgumentException] {
      Graph(fc1, relu1)
    }
  }

  "Graph init" should "be successful when inputs node are same with outputs node" in {
    val fc1 = Linear(4, 2).apply()
    val graph = Graph(fc1, fc1)

    val inputData = Tensor(4, 4)
    fc1.element.parameters()._1(1).zero() // bias is set to 0
    graph.forward(inputData) should be((inputData * fc1.element.parameters()._1(0).t()))
  }

  "Graph init" should "throw exceptions when some inputs are ignored" in {
    val fc1 = Linear(4, 2).apply()
    val fc2 = Linear(4, 2).apply()
    val output = CAddTable().apply(fc1, fc2)

    intercept[IllegalArgumentException] {
      Graph(fc1, output)
    }
  }

  "Graph init" should "be successful output are ignored" in {
    val fc1 = Linear(4, 2).apply()
    val fc2 = Linear(4, 2).apply()
    val cadd = CAddTable().apply(fc1, fc2)
    val output1 = ReLU().apply(cadd)
    val output2 = ReLU().apply(cadd)

    val graph = Graph(Array(fc1, fc2), Array(output1))
    fc1.element.getParameters()._1.apply1(_ => 1.0f)
    fc2.element.getParameters()._1.apply1(_ => 1.0f)
    val output = graph.forward(T(Tensor(T(0.1f, 0.2f, -0.3f, -0.4f)),
      Tensor(T(0.5f, 0.4f, -0.2f, -0.1f))))
    output should be(Tensor(T(2.2f, 2.2f)))
  }

  "Graph init" should "throw exceptions when input a tensor while a table is required" in {
    val fc1 = Linear(4, 2).apply()
    val fc2 = Linear(4, 2).apply()
    val cadd = CAddTable().apply(fc1, fc2)
    val output1 = ReLU().apply(cadd)
    val output2 = ReLU().apply(cadd)

    val graph = Graph(Array(fc1, fc2), Array(output1, output2))
    intercept[IllegalArgumentException] {
      graph.forward(Tensor(T(0.1f, 0.2f, -0.3f, -0.4f)))
    }
  }

  "Graph init" should "throw exceptions when inputs has pre-nodes" in {
    val fc1 = Linear(4, 2).apply()
    val fc2 = Linear(4, 2).apply()
    val tanh1 = Tanh().apply(fc1)
    val tanh2 = Tanh().apply(fc2)

    val cadd = CAddTable().apply(tanh1, tanh2)
    val output1 = ReLU().apply(cadd)
    val output2 = ReLU().apply(cadd)

    intercept[IllegalArgumentException] {
      Graph(Array(tanh1, tanh2), Array(output1, output2))
    }
  }

  "Graph init" should "throw exceptions when inputs has nothing to do with the graph but same " +
    "number with the roots node in the graph" in {
    val fc1 = Linear(4, 2).apply()
    val fc2 = Linear(4, 2).apply()
    val fc3 = Linear(4, 2).apply()
    val fc4 = Linear(4, 2).apply()
    val tanh1 = Tanh().apply(fc1)
    val tanh2 = Tanh().apply(fc2)

    val cadd = CAddTable().apply(tanh1, tanh2)
    val output1 = ReLU().apply(cadd)
    val output2 = ReLU().apply(cadd)

    intercept[IllegalArgumentException] {
      Graph(Array(fc3, fc4), Array(output1, output2))
    }
  }

  "Graph forward" should "be successful" in {
    val fc1 = Linear(4, 2).apply()
    val fc2 = Linear(4, 2).apply()
    val cadd = CAddTable().apply(fc1, fc2)
    val output1 = ReLU().apply(cadd)
    val output2 = Threshold(10.0).apply(cadd)

    val graph = Graph(Array(fc1, fc2), Array(output1, output2))
    fc1.element.getParameters()._1.apply1(_ => 1.0f)
    fc2.element.getParameters()._1.apply1(_ => 1.0f)
    val output = graph.forward(T(Tensor(T(0.1f, 0.2f, -0.3f, -0.4f)),
      Tensor(T(0.5f, 0.4f, -0.2f, -0.1f))))
    output should be(T(Tensor(T(2.2f, 2.2f)), Tensor(T(.0f, .0f))))
  }

  "Graph forward" should "throw exceptions when input a table while a tensor is required" in {
    val fc1 = Linear(4, 2).apply()
    val output1 = ReLU().apply(fc1)

    val graph = Graph(Array(fc1), Array(output1))

    intercept[IllegalArgumentException] {
      graph.forward(T(Tensor(T(0.1f, 0.2f, -0.3f, -0.4f)),
        Tensor(T(0.5f, 0.4f, -0.2f, -0.1f))))
    }
  }

  "Graph forward" should "be successful when first node accept multiple tensors input" in {
    val input1 = Input()
    val input2 = Input()
    val cadd = CAddTable().apply(input1, input2)
    val graph = Graph(Array(input1, input2), cadd)
    val output = graph.forward(T(Tensor(T(0.1f, 0.2f, -0.3f, -0.4f)),
      Tensor(T(0.5f, 0.4f, -0.2f, -0.1f))))
    output should be(Tensor(T(0.6f, 0.6f, -0.5f, -0.5f)))
  }

  "Graph forward" should "be successful when exchange input order" in {
    val fc1 = Linear(4, 2).apply()
    val fc2 = Linear(4, 2).apply()
    val cadd = CAddTable().apply(fc1, fc2)
    val output1 = ReLU().apply(cadd)
    val output2 = Threshold(10.0).apply(cadd)

    val graph = Graph(Array(fc2, fc1), Array(output1, output2))
    fc1.element.getParameters()._1.apply1(_ => 1.0f)
    fc2.element.getParameters()._1.apply1(_ => 2.0f)
    val output = graph.forward(T(Tensor(T(0.1f, 0.2f, -0.3f, -0.4f)),
      Tensor(T(0.5f, 0.4f, -0.2f, -0.1f))))
    output should be(T(Tensor(T(2.8f, 2.8f)), Tensor(T(0.0f, 0.0f))))
  }

  "Graph forward" should "be successful when paths has different length" in {
    val fc1 = Linear(4, 2).apply()
    val thd1 = Threshold(-10.0).apply(fc1)
    val thd2 = Threshold(-10.0).apply(thd1)
    val thd3 = Threshold(-10.0).apply(thd2)
    val thd4 = Threshold(-10.0).apply(thd3)
    val fc2 = Linear(4, 2).apply()
    val cadd = CAddTable().apply(thd4, fc2)
    val output1 = ReLU().apply(cadd)
    val output2 = Threshold(10.0).apply(cadd)

    val graph = Graph(Array(fc1, fc2), Array(output1, output2))
    fc1.element.getParameters()._1.apply1(_ => 1.0f)
    fc2.element.getParameters()._1.apply1(_ => 1.0f)
    val output = graph.forward(T(Tensor(T(0.1f, 0.2f, -0.3f, -0.4f)),
      Tensor(T(0.5f, 0.4f, -0.2f, -0.1f))))
    output should be(T(Tensor(T(2.2f, 2.2f)), Tensor(T(.0f, .0f))))
  }

  "Graph forward" should "be successful when exchange output order" in {
    val fc1 = Linear(4, 2).apply()
    val fc2 = Linear(4, 2).apply()
    val cadd = CAddTable().apply(fc1, fc2)
    val output1 = ReLU().apply(cadd)
    val output2 = Threshold(10.0).apply(cadd)

    val graph = Graph(Array(fc1, fc2), Array(output2, output1))
    fc1.element.getParameters()._1.apply1(_ => 1.0f)
    fc2.element.getParameters()._1.apply1(_ => 2.0f)
    val output = graph.forward(T(Tensor(T(0.1f, 0.2f, -0.3f, -0.4f)),
      Tensor(T(0.5f, 0.4f, -0.2f, -0.1f))))
    output should be(T(Tensor(T(0.0f, 0.0f)), Tensor(T(3.8f, 3.8f))))
  }

  "Graph backward" should "be successful" in {
    val fc1 = Linear(4, 2).apply()
    val fc2 = Linear(4, 2).apply()
    val cadd = CAddTable().apply(fc1, fc2)
    val output1 = ReLU().apply(cadd)
    val output2 = Threshold(10.0).apply(cadd)

    val graph = Graph(Array(fc1, fc2), Array(output1, output2))
    fc1.element.getParameters()._1.apply1(_ => 1.0f)
    fc2.element.getParameters()._1.apply1(_ => 2.0f)
    val output = graph.forward(T(Tensor(T(0.1f, 0.2f, -0.3f, -0.4f)),
      Tensor(T(0.5f, 0.4f, -0.2f, -0.1f))))
    val gradInput = graph.backward(T(Tensor(T(0.1f, 0.2f, -0.3f, -0.4f)),
      Tensor(T(0.5f, 0.4f, -0.2f, -0.1f))), T(Tensor(T(1.0f, 2.0f)), Tensor(T(3.0f, 4.0f))))
    gradInput should be(T(Tensor(T(3.0f, 3.0f, 3.0f, 3.0f)),
      Tensor(T(6.0f, 6.0f, 6.0f, 6.0f))))
    fc1.element.parameters()._2(0) should be(Tensor(T(T(0.1f, 0.2f, -0.3f, -0.4f),
      T(0.2f, 0.4f, -0.6f, -0.8f))))
    fc1.element.parameters()._2(1) should be(Tensor(T(1.0f, 2.0f)))
    fc2.element.parameters()._2(0) should be(Tensor(T(T(0.5f, 0.4f, -0.2f, -0.1f),
      T(1.0f, 0.8f, -0.4f, -0.2f))))
    fc2.element.parameters()._2(1) should be(Tensor(T(1.0f, 2.0f)))
  }

  "Graph backward" should "be successful when first node accept multiple tensors input" in {
    val input1 = Input()
    val input2 = Input()
    val cadd = CAddTable().apply(input1, input2)
    val graph = Graph(Array(input1, input2), cadd)
    val output = graph.forward(T(Tensor(T(0.1f, 0.2f, -0.3f, -0.4f)),
      Tensor(T(0.5f, 0.4f, -0.2f, -0.1f))))
    output should be(Tensor(T(0.6f, 0.6f, -0.5f, -0.5f)))
    val gradient = graph.backward(T(Tensor(T(0.1f, 0.2f, -0.3f, -0.4f)),
      Tensor(T(0.5f, 0.4f, -0.2f, -0.1f))), Tensor(T(0.1f, 0.1f, 0.1f, 0.1f)))
    gradient should be(T(Tensor(T(0.1f, 0.1f, 0.1f, 0.1f)), Tensor(T(0.1f, 0.1f, 0.1f, 0.1f))))
  }

  "Graph backward" should "be successful when paths have different length" in {
    val fc1 = Linear(4, 2).apply()
    val thd1 = Threshold(-10.0).apply(fc1)
    val thd2 = Threshold(-10.0).apply(thd1)
    val thd3 = Threshold(-10.0).apply(thd2)
    val thd4 = Threshold(-10.0).apply(thd3)
    val fc2 = Linear(4, 2).apply()
    val cadd = CAddTable().apply(thd4, fc2)
    val output1 = ReLU().apply(cadd)
    val output2 = Threshold(10.0).apply(cadd)

    val graph = Graph(Array(fc1, fc2), Array(output1, output2))
    fc1.element.getParameters()._1.apply1(_ => 1.0f)
    fc2.element.getParameters()._1.apply1(_ => 2.0f)
    val output = graph.forward(T(Tensor(T(0.1f, 0.2f, -0.3f, -0.4f)),
      Tensor(T(0.5f, 0.4f, -0.2f, -0.1f))))
    val gradInput = graph.backward(T(Tensor(T(0.1f, 0.2f, -0.3f, -0.4f)),
      Tensor(T(0.5f, 0.4f, -0.2f, -0.1f))), T(Tensor(T(1.0f, 2.0f)), Tensor(T(3.0f, 4.0f))))
    gradInput should be(T(Tensor(T(3.0f, 3.0f, 3.0f, 3.0f)),
      Tensor(T(6.0f, 6.0f, 6.0f, 6.0f))))
    fc1.element.parameters()._2(0) should be(Tensor(T(T(0.1f, 0.2f, -0.3f, -0.4f),
      T(0.2f, 0.4f, -0.6f, -0.8f))))
    fc1.element.parameters()._2(1) should be(Tensor(T(1.0f, 2.0f)))
    fc2.element.parameters()._2(0) should be(Tensor(T(T(0.5f, 0.4f, -0.2f, -0.1f),
      T(1.0f, 0.8f, -0.4f, -0.2f))))
    fc2.element.parameters()._2(1) should be(Tensor(T(1.0f, 2.0f)))
  }

  "Graph backward" should "be successful when exchange input order" in {
    val fc1 = Linear(4, 2).apply()
    val fc2 = Linear(4, 2).apply()
    val cadd = CAddTable().apply(fc1, fc2)
    val output1 = ReLU().apply(cadd)
    val output2 = Threshold(10.0).apply(cadd)

    val graph = Graph(Array(fc2, fc1), Array(output1, output2))
    fc1.element.getParameters()._1.apply1(_ => 1.0f)
    fc2.element.getParameters()._1.apply1(_ => 2.0f)
    val output = graph.forward(T(Tensor(T(0.1f, 0.2f, -0.3f, -0.4f)),
      Tensor(T(0.5f, 0.4f, -0.2f, -0.1f))))
    val gradInput = graph.backward(T(Tensor(T(0.1f, 0.2f, -0.3f, -0.4f)),
      Tensor(T(0.5f, 0.4f, -0.2f, -0.1f))), T(Tensor(T(1.0f, 2.0f)), Tensor(T(3.0f, 4.0f))))
    gradInput should be(T(Tensor(T(6.0f, 6.0f, 6.0f, 6.0f)), Tensor(T(3.0f, 3.0f, 3.0f, 3.0f))))
    fc1.element.parameters()._2(0) should be(Tensor(T(T(0.5f, 0.4f, -0.2f, -0.1f),
      T(1.0f, 0.8f, -0.4f, -0.2f))))
    fc1.element.parameters()._2(1) should be(Tensor(T(1.0f, 2.0f)))
    fc2.element.parameters()._2(0) should be(Tensor(T(T(0.1f, 0.2f, -0.3f, -0.4f),
      T(0.2f, 0.4f, -0.6f, -0.8f))))
    fc2.element.parameters()._2(1) should be(Tensor(T(1.0f, 2.0f)))
  }

  "Graph backward" should "be successful when exchange output order" in {
    val fc1 = Linear(4, 2).apply()
    val fc2 = Linear(4, 2).apply()
    val cadd = CAddTable().apply(fc1, fc2)
    val output1 = ReLU().apply(cadd)
    val output2 = Threshold(10.0).apply(cadd)

    val graph = Graph(Array(fc1, fc2), Array(output2, output1))
    fc1.element.getParameters()._1.apply1(_ => 1.0f)
    fc2.element.getParameters()._1.apply1(_ => 2.0f)
    val output = graph.forward(T(Tensor(T(0.1f, 0.2f, -0.3f, -0.4f)),
      Tensor(T(0.5f, 0.4f, -0.2f, -0.1f))))
    val gradInput = graph.backward(T(Tensor(T(0.1f, 0.2f, -0.3f, -0.4f)),
      Tensor(T(0.5f, 0.4f, -0.2f, -0.1f))), T(Tensor(T(1.0f, 2.0f)), Tensor(T(3.0f, 4.0f))))
    gradInput should be(T(Tensor(T(7.0f, 7.0f, 7.0f, 7.0f)), Tensor(T(14.0f, 14.0f, 14.0f, 14.0f))))
    fc1.element.parameters()._2(0) should be(Tensor(T(T(0.3f, 0.6f, -0.9f, -1.2f),
      T(0.4f, 0.8f, -1.2f, -1.6f))))
    fc1.element.parameters()._2(1) should be(Tensor(T(3.0f, 4.0f)))
    fc2.element.parameters()._2(0) should be(Tensor(T(T(1.5f, 1.2f, -0.6f, -0.3f),
      T(2.0f, 1.6f, -0.8f, -0.4f))))
    fc2.element.parameters()._2(1) should be(Tensor(T(3.0f, 4.0f)))
  }

  "lenet" should "be same with sequential model" in {
    RandomGenerator.RNG.setSeed(1000)
    val seqModel = Sequential().add(Reshape(Array(1, 28, 28)))
      .add(SpatialConvolution(1, 6, 5, 5).setName("conv1_5x5"))
      .add(Tanh())
      .add(SpatialMaxPooling(2, 2, 2, 2))
      .add(Tanh())
      .add(SpatialConvolution(6, 12, 5, 5).setName("conv2_5x5"))
      .add(SpatialMaxPooling(2, 2, 2, 2))
      .add(Reshape(Array(12 * 4 * 4)))
      .add(Linear(12 * 4 * 4, 100).setName("fc1"))
      .add(Tanh())
      .add(Linear(100, 10).setName("fc2"))
      .add(LogSoftMax())

    RandomGenerator.RNG.setSeed(1000)
    val input = Reshape(Array(1, 28, 28)).apply()
    val conv1 = SpatialConvolution(1, 6, 5, 5).apply(input)
    val tanh1 = Tanh().apply(conv1)
    val pool1 = SpatialMaxPooling(2, 2, 2, 2).apply(tanh1)
    val tanh2 = Tanh().apply(pool1)
    val conv2 = SpatialConvolution(6, 12, 5, 5).apply(tanh2)
    val pool2 = SpatialMaxPooling(2, 2, 2, 2).apply(conv2)
    val reshape = Reshape(Array(12 * 4 * 4)).apply(pool2)
    val fc1 = Linear(12 * 4 * 4, 100).apply(reshape)
    val tanh3 = Tanh().apply(fc1)
    val fc2 = Linear(100, 10).apply(tanh3)
    val output = LogSoftMax().apply(fc2)

    val funcModel = Graph(input, output)

    val inputData = Tensor(4, 28 * 28).rand()
    val outputData1 = seqModel.forward(inputData) // warm up
    var start = System.nanoTime()
    seqModel.forward(inputData)
    println(s"seq model forward time is ${(System.nanoTime() - start) / 1e6}ms")
    start = System.nanoTime()
    val outputData2 = funcModel.forward(inputData)
    println(s"funcModel model forward time is ${(System.nanoTime() - start) / 1e6}ms")

    outputData1 should be(outputData2)

    val gradient = Tensor(4, 10).rand()
    start = System.nanoTime()
    val gradientBP1 = seqModel.backward(inputData, gradient)
    println(s"seq model backward time is ${(System.nanoTime() - start) / 1e6}ms")
    start = System.nanoTime()
    val gradientBP2 = funcModel.backward(inputData, gradient)
    println(s"funcModel model backward time is ${(System.nanoTime() - start) / 1e6}ms")

    gradientBP1 should be(gradientBP2)
    seqModel.getParameters()._2 should be(funcModel.getParameters()._2)
  }

  "shift" should "be correct" in {
    val node = Reshape(Array(1, 28, 28)).apply()
    val test = Graph(node, node)
    test.shift(Array(1, 2, 3, 4), 1, 1) should be(Array(1, 2, 3, 4))
    test.shift(Array(1, 2, 3, 4), 1, 3) should be(Array(1, 3, 4, 2))
    test.shift(Array(1, 2, 3, 4), 3, 1) should be(Array(1, 4, 2, 3))
  }

  "ResNet-18 basic block shortcut type A" should "be correct" in {
    RandomGenerator.RNG.setSeed(1000)
    val seqModel = ModelUntils.ResNet.basicBlockSeq(16, 16, 1, "A")
    RandomGenerator.RNG.setSeed(1000)
    val input = Input()
    val output = ModelUntils.ResNet.basicBlockSeq(16, 16, 1, "A")(input)
    val funcModel = Graph(input, output)

    println(seqModel)
    val inputData = Tensor(4, 16, 32, 32).rand()
    var start = System.nanoTime()
    val output1 = seqModel.forward(inputData)
    println(s"seq model forward time is ${(System.nanoTime() - start) / 1e6}ms")

    start = System.nanoTime()
    val output2 = funcModel.forward(inputData)
    println(s"func model forward time is ${(System.nanoTime() - start) / 1e6}ms")

    output1 should be(output2)

    val gradients = Tensor(4, 16, 32, 32).rand()
    start = System.nanoTime()
    val gradients1 = seqModel.backward(inputData, gradients)
    println(s"seq model backward time is ${(System.nanoTime() - start) / 1e6}ms")
    start = System.nanoTime()
    val gradients2 = funcModel.backward(inputData, gradients)
    println(s"func model backward time is ${(System.nanoTime() - start) / 1e6}ms")

    gradients1 should be(gradients2)
    seqModel.getParameters()._2 should be(funcModel.getParameters()._2)
  }

  "ResNet-18 basic block shortcut type C" should "be correct" in {
    RandomGenerator.RNG.setSeed(1000)
    val seqModel = ModelUntils.ResNet.basicBlockSeq(16, 16, 1, "C")
    RandomGenerator.RNG.setSeed(1000)
    val input = Input()
    val output = ModelUntils.ResNet.basicBlockSeq(16, 16, 1, "C")(input)
    val funcModel = Graph(input, output)

    println(seqModel)
    val inputData = Tensor(4, 16, 32, 32).rand()
    var start = System.nanoTime()
    val output1 = seqModel.forward(inputData)
    println(s"seq model forward time is ${(System.nanoTime() - start) / 1e6}ms")

    start = System.nanoTime()
    val output2 = funcModel.forward(inputData)
    println(s"func model forward time is ${(System.nanoTime() - start) / 1e6}ms")

    output1 should be(output2)

    val gradients = Tensor(4, 16, 32, 32).rand()
    start = System.nanoTime()
    val gradients1 = seqModel.backward(inputData, gradients)
    println(s"seq model backward time is ${(System.nanoTime() - start) / 1e6}ms")
    start = System.nanoTime()
    val gradients2 = funcModel.backward(inputData, gradients)
    println(s"func model backward time is ${(System.nanoTime() - start) / 1e6}ms")

    gradients1 should be(gradients2)
    seqModel.getParameters()._2 should be(funcModel.getParameters()._2)
  }

  "InceptionV1 block" should "be correct" in {
    RandomGenerator.RNG.setSeed(1000)
    val seqModel = ModelUntils.Inception.inceptionLayerV1Seq(
      2, T(T(4), T(96, 128), T(16, 32), T(32)))

    RandomGenerator.RNG.setSeed(1000)
    val input = Input()
    val output = ModelUntils.Inception.inceptionLayerV1Func(
      2, T(T(4), T(96, 128), T(16, 32), T(32)))(input)
    val funcModel = Graph(input, output)

    println(seqModel)
    val inputData = Tensor(1, 2, 4, 4).rand()
    var start = System.nanoTime()
    val output1 = seqModel.forward(inputData)
    println(s"seq model forward time is ${(System.nanoTime() - start) / 1e6}ms")

    start = System.nanoTime()
    val output2 = funcModel.forward(inputData)
    println(s"func model forward time is ${(System.nanoTime() - start) / 1e6}ms")

    output1 should be(output2)
    val gradient = Tensor(1, 256, 4, 4).rand()
    start = System.nanoTime()
    val gradient1 = seqModel.backward(inputData, gradient)
    println(s"seq model backward time is ${(System.nanoTime() - start) / 1e6}ms")

    start = System.nanoTime()
    val gradient2 = funcModel.backward(inputData, gradient)

    println(s"func model backward time is ${(System.nanoTime() - start) / 1e6}ms")

    gradient1 should be(gradient2)

    seqModel.getParametersTable()[Table]("conv1x1")[Tensor[Float]]("gradWeight") should be(
      funcModel.getParametersTable()[Table]("conv1x1")[Tensor[Float]]("gradWeight")
    )

    seqModel.getParametersTable()[Table]("conv3x3_1")[Tensor[Float]]("gradWeight") should be(
      funcModel.getParametersTable()[Table]("conv3x3_1")[Tensor[Float]]("gradWeight")
    )

    seqModel.getParametersTable()[Table]("conv3x3_2")[Tensor[Float]]("gradWeight") should be(
      funcModel.getParametersTable()[Table]("conv3x3_2")[Tensor[Float]]("gradWeight")
    )

    seqModel.getParametersTable()[Table]("conv5x5_1")[Tensor[Float]]("gradWeight") should be(
      funcModel.getParametersTable()[Table]("conv5x5_1")[Tensor[Float]]("gradWeight")
    )

    seqModel.getParametersTable()[Table]("conv5x5_2")[Tensor[Float]]("gradWeight") should be(
      funcModel.getParametersTable()[Table]("conv5x5_2")[Tensor[Float]]("gradWeight")
    )

    seqModel.getParametersTable()[Table]("pool_conv")[Tensor[Float]]("gradWeight") should be(
      funcModel.getParametersTable()[Table]("pool_conv")[Tensor[Float]]("gradWeight")
    )
  }
}

object ModelUntils {
  object Inception {
    def inceptionLayerV1Func(inputSize: Int, config: Table)(input : ModuleNode[Float])
    : ModuleNode[Float] = {
      val conv1x1 = SpatialConvolution(inputSize, config[Table](1)(1), 1, 1, 1, 1)
        .setName("conv1x1").apply(input)
      val relu1x1 = ReLU(true).apply(conv1x1)

      val conv3x3_1 = SpatialConvolution(inputSize, config[Table](2)(1), 1, 1, 1, 1)
        .setName("conv3x3_1").apply(input)
      val relu3x3_1 = ReLU(true).apply(conv3x3_1)
      val conv3x3_2 = SpatialConvolution(
        config[Table](2)(1), config[Table](2)(2), 3, 3, 1, 1, 1, 1)
        .setName("conv3x3_2").apply(relu3x3_1)
      val relu3x3_2 = ReLU(true).apply(conv3x3_2)

      val conv5x5_1 = SpatialConvolution(inputSize, config[Table](3)(1), 1, 1, 1, 1)
        .setName("conv5x5_1").apply(input)
      val relu5x5_1 = ReLU(true).apply(conv5x5_1)
      val conv5x5_2 = SpatialConvolution(
        config[Table](3)(1), config[Table](3)(2), 5, 5, 1, 1, 2, 2)
        .setName("conv5x5_2").apply(relu5x5_1)
      val relu5x5_2 = ReLU(true).apply(conv5x5_2)

      val pool = SpatialMaxPooling(3, 3, 1, 1, 1, 1).ceil()
        .setName("pool").apply(input)
      val convPool = SpatialConvolution(inputSize, config[Table](4)(1), 1, 1, 1, 1)
        .setName("pool_conv").apply(pool)
      val reluPool = ReLU(true).apply(convPool)

      JoinTable(2, 4).apply(relu1x1, relu3x3_2, relu5x5_2, reluPool)
    }
    def inceptionLayerV1Seq(inputSize: Int, config: Table) : Module[Float] = {
      val concat = Concat(2)
      val conv1 = Sequential()
      conv1.add(SpatialConvolution(inputSize, config[Table](1)(1), 1, 1, 1, 1)
        .setName("conv1x1"))
      conv1.add(ReLU(true))
      concat.add(conv1)
      val conv3 = Sequential()
      conv3.add(SpatialConvolution(inputSize, config[Table](2)(1), 1, 1, 1, 1)
        .setName("conv3x3_1"))
      conv3.add(ReLU(true))
      conv3.add(SpatialConvolution(config[Table](2)(1), config[Table](2)(2), 3, 3, 1, 1, 1, 1)
        .setName("conv3x3_2"))
      conv3.add(ReLU(true))
      concat.add(conv3)
      val conv5 = Sequential()
      conv5.add(SpatialConvolution(inputSize, config[Table](3)(1), 1, 1, 1, 1)
        .setName("conv5x5_1"))
      conv5.add(ReLU(true))
      conv5.add(SpatialConvolution(config[Table](3)(1), config[Table](3)(2), 5, 5, 1, 1, 2, 2)
        .setName("conv5x5_2"))
      conv5.add(ReLU(true))
      concat.add(conv5)
      val pool = Sequential()
      pool.add(SpatialMaxPooling(3, 3, 1, 1, 1, 1).ceil()
        .setName("pool"))
      pool.add(SpatialConvolution(inputSize, config[Table](4)(1), 1, 1, 1, 1).setName("pool_conv"))
      pool.add(ReLU(true))
      concat.add(pool)
      concat
    }
  }
  object ResNet {
    def basicBlockFunc(nInputPlane: Int, n: Int, stride: Int, shortcutType : String)(
      input : ModuleNode[Float]) : ModuleNode[Float] = {
      val conv1 = SpatialConvolution(nInputPlane, n, 3, 3, stride, stride, 1, 1).apply(input)
      val bn1 = SpatialBatchNormalization(n).apply(conv1)
      val relu1 = ReLU(true).apply(bn1)
      val conv2 = SpatialConvolution(n, n, 3, 3, 1, 1, 1, 1).apply(relu1)
      val bn2 = SpatialBatchNormalization(n).apply(conv2)
      val shortcut = shortcutFunc(nInputPlane, n, stride, shortcutType)(input)
      val add = CAddTable(true).apply(bn2, shortcut)
      val output = ReLU(true).apply(add)
      output
    }

    def basicBlockSeq(nInputPlane: Int, n: Int, stride: Int, shortcutType : String)
    : Module[Float] = {
      val s = Sequential()
      s.add(SpatialConvolution(nInputPlane, n, 3, 3, stride, stride, 1, 1))
      s.add(SpatialBatchNormalization(n))
      s.add(ReLU(true))
      s.add(SpatialConvolution(n, n, 3, 3, 1, 1, 1, 1))
      s.add(SpatialBatchNormalization(n))

      Sequential()
        .add(ConcatTable()
          .add(s)
          .add(shortcutSeq(nInputPlane, n, stride, shortcutType)))
        .add(CAddTable(true))
        .add(ReLU(true))
    }

    def shortcutFunc(nInputPlane: Int, nOutputPlane: Int, stride: Int,
                     shortcutType : String)(input : ModuleNode[Float]) : ModuleNode[Float] = {
      val useConv = shortcutType == "C" || (shortcutType == "B" && nInputPlane != nOutputPlane)

      if (useConv) {
        val conv1 = SpatialConvolution(nInputPlane, nOutputPlane, 1, 1, stride, stride).apply(input)
        val bn1 = SpatialBatchNormalization(nOutputPlane).apply(conv1)
        bn1
      } else if (nInputPlane != nOutputPlane) {
        val pool1 = SpatialAveragePooling(1, 1, stride, stride).apply(input)
        val mul1 = MulConstant(0f).apply(pool1)
        val concat = JoinTable(2, 3).apply(pool1, mul1)
        concat
      } else {
        input
      }
    }
    def shortcutSeq(nInputPlane: Int, nOutputPlane: Int, stride: Int, shortcutType : String)
    : Module[Float] = {
      val useConv = shortcutType == "C" || (shortcutType == "B" && nInputPlane != nOutputPlane)

      if (useConv) {
        Sequential()
          .add(SpatialConvolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
          .add(SpatialBatchNormalization(nOutputPlane))
      } else if (nInputPlane != nOutputPlane) {
        Sequential()
          .add(SpatialAveragePooling(1, 1, stride, stride))
          .add(Concat(2)
            .add(Identity())
            .add(MulConstant(0f)))
      } else {
        Identity()
      }
    }
  }
}
