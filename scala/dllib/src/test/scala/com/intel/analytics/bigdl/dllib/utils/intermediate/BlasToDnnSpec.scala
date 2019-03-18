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

package com.intel.analytics.bigdl.utils.intermediate

import breeze.numerics._
import com.intel.analytics.bigdl.example.loadmodel.AlexNet
import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.models.inception.{Inception_Layer_v1, Inception_v1_NoAuxClassifier}
import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.models.resnet.ResNet
import com.intel.analytics.bigdl.models.resnet.ResNet.{DatasetType, ShortcutType}
import com.intel.analytics.bigdl.models.vgg.Vgg_16
import com.intel.analytics.bigdl.nn.{Module, StaticGraph}
import com.intel.analytics.bigdl.nn
import com.intel.analytics.bigdl.nn.mkldnn.{DnnGraph, Equivalent}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils._

import scala.util.Random

class BlasToDnnSpec extends BigDLSpecHelper {
  override def doBefore(): Unit = {
    System.setProperty("bigdl.engineType", "mkldnn")
  }

  override def doAfter(): Unit = {
    System.clearProperty("bigdl.engineType")
  }
  "vgg16 blas to dnn" should "work properly" in {
    val batchSize = 2
    val classNum = 1000
    RandomGenerator.RNG.setSeed(1000)
    val input = Tensor[Float](Array(batchSize, 3, 224, 224)).apply1(_ =>
      RandomGenerator.RNG.uniform(0.1, 1.0).toFloat)
    val gradOutput = Tensor[Float](batchSize, classNum).apply1(_ =>
      RandomGenerator.RNG.uniform(1.0, 1000.0).toFloat)

    val blas = Vgg_16.graph(classNum, false).asInstanceOf[StaticGraph[Float]]
    blas.setInputFormats(Seq(Memory.Format.nchw))
    blas.setOutputFormats(Seq(Memory.Format.nc))
    val irBlas = blas.cloneModule().toIRgraph()

    val outBlas = blas.forward(input).toTensor[Float]
    val gradInputBlas = blas.backward(input, gradOutput).toTensor[Float]

    val outDnn = irBlas.forward(input).toTensor[Float]
    val gradInputDnn = irBlas.backward(input, gradOutput).toTensor[Float]

    Equivalent.nearequals(outDnn, outBlas, 1e-6) should be(true)
    Equivalent.nearequals(gradInputDnn, gradInputBlas, 1e-4) should be(true)
  }

  "lenet5 blas to dnn" should "work properly" in {
    val batchSize = 2
    val seed = 1
    val inputFormat = Memory.Format.nchw
    val inputShape = Array(batchSize, 1, 28, 28)

    val input = Tensor[Float](inputShape).rand()
    val gradOutput = Tensor[Float](batchSize, 10).rand()

    val blas = LeNet5.graph(10).asInstanceOf[StaticGraph[Float]]
    blas.setInputFormats(Seq(Memory.Format.nchw))
    blas.setOutputFormats(Seq(Memory.Format.nc))
    val irBlas = blas.cloneModule().toIRgraph()

    val outBlas = blas.forward(input).toTensor[Float]
    val gradInputBlas = blas.backward(input, gradOutput).toTensor[Float]

    val outDnn = irBlas.forward(input).toTensor[Float]
    val gradInputDnn = irBlas.backward(input, gradOutput).toTensor[Float]

    Equivalent.nearequals(outDnn, outBlas, 1e-6) should be(true)
    Equivalent.nearequals(gradInputDnn, gradInputBlas, 1e-6) should be(true)
  }

  "inception_v1 blas to dnn" should "work properly" in {
    val batchSize = 2
    val classNum = 1000
    RandomGenerator.RNG.setSeed(1)

    val input = Tensor[Float](Array(batchSize, 3, 224, 224)).apply1(_ =>
      RandomGenerator.RNG.uniform(0.1, 1.0).toFloat)
    val gradOutput = Tensor[Float](batchSize, classNum).apply1(_ =>
      RandomGenerator.RNG.uniform(1, 10).toFloat)

    val blas = Inception_v1_NoAuxClassifier.graph(classNum, false).asInstanceOf[StaticGraph[Float]]
    blas.setInputFormats(Seq(Memory.Format.nchw))
    blas.setOutputFormats(Seq(Memory.Format.nc))
    val irBlas = blas.cloneModule().toIRgraph()

    val outBlas = blas.forward(input).toTensor[Float]
    val gradInputBlas = blas.backward(input, gradOutput).toTensor[Float]

    val outDnn = irBlas.forward(input).toTensor[Float]
    val gradInputDnn = irBlas.backward(input, gradOutput).toTensor[Float]

    Equivalent.nearequals(outDnn, outBlas, 1e-6) should be(true)
  }

  "resnet50 blas to dnn" should "work properly" in {
    val batchSize = 2
    val classNum = 1000
    RandomGenerator.RNG.setSeed(1000)
    val input = Tensor[Float](Array(batchSize, 3, 224, 224)).apply1(_ =>
      RandomGenerator.RNG.uniform(0.1, 1.0).toFloat)
    var gradOutput = Tensor[Float](batchSize, classNum).apply1(_ =>
      RandomGenerator.RNG.uniform(1.0, 1000.0).toFloat)

    val blas = ResNet.graph(classNum,
      T("shortcutType" -> ShortcutType.B, "depth" -> 50,
        "optnet" -> false, "dataset" -> DatasetType.ImageNet)).asInstanceOf[StaticGraph[Float]]
    val irBlas = blas.toIRgraph()

    irBlas.build()
    val outBlas = blas.forward(input).toTensor[Float]
    val outDnn = irBlas.forward(input).toTensor[Float]


    gradOutput.resizeAs(outBlas).apply1(_ =>
      RandomGenerator.RNG.uniform(1.0, 1000.0).toFloat)

    val gradInputBlas = blas.backward(input, gradOutput).toTensor[Float]

    val gradInputDnn = irBlas.backward(input, gradOutput).toTensor[Float]
    val gradInputTensor = Tensor[Float]().resize(gradInputDnn.size()).copy(gradInputDnn)

    Equivalent.nearequals(outDnn, outBlas, 1e-6) should be(true)
  }

  "resnet50 dnn to blas" should "work properly" in {
    val batchSize = 2
    val classNum = 1000
    RandomGenerator.RNG.setSeed(1000)
    val blas = ResNet.graph(classNum,
      T("shortcutType" -> ShortcutType.B, "depth" -> 50,
        "optnet" -> false, "dataset" -> DatasetType.ImageNet)).asInstanceOf[StaticGraph[Float]]
    val irBlas = blas.toIRgraph()

    for (i <- 0 to 3) {
      val input = Tensor[Float](2, 3, 224, 224).rand()
      val gradOutput = Tensor[Float](2, 1000).rand()
      irBlas.training()
      irBlas.forward(input)
      irBlas.backward(input, gradOutput)
    }
    val input = Tensor[Float](2, 3, 224, 224).rand()
    irBlas.evaluate()
    irBlas.forward(input)

    val p1 = blas.getParameters()
    val p2 = irBlas.getParameters()
    p1._1.copy(p2._1)
    p1._2.copy(p2._2)
    blas.setExtraParameter(irBlas.getExtraParameter())

    val in = Tensor[Float](2, 3, 224, 224).rand()
    blas.evaluate()
    irBlas.evaluate()

    val out1 = blas.forward(in).toTensor[Float]
    val out2 = irBlas.forward(in).toTensor[Float]

    Equivalent.getunequals(out1, out2, 1e-4) should be(true)
  }
}
