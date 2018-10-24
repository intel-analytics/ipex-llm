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

import breeze.linalg.Axis._1
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.models.resnet.ResNet.{DatasetType, ShortcutType}
import com.intel.analytics.bigdl.nn.mkldnn.Phase.TrainingPhase
import com.intel.analytics.bigdl.nn.{Module => _, _}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{T, Table}
import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.models.resnet


class DnnGraphSpec extends FlatSpec with Matchers {

  "Dnn vgg16 graph model" should "be correct" in {
    val batchSize = 2
    val seed = 1
    val inputFormat = Memory.Format.nchw
    val inputShape = Array(batchSize, 3, 224, 224)

    RNG.setSeed(seed)
    val graphModel = models.Vgg_16.graph(batchSize, 1000, false)
    RNG.setSeed(seed)
    val dnnModle = models.Vgg_16(batchSize, 1000, false)

    // graphModel.asInstanceOf[DnnGraph].compile(TrainingPhase)
    dnnModle.compile(TrainingPhase)

    val input = Tensor[Float](inputShape).rand()
    val gradOutput = Tensor[Float](batchSize, 1000).rand()

    for (i <- 0 to 2) {
      graphModel.forward(input)
      dnnModle.forward(input)

      graphModel.backward(input, gradOutput)
      dnnModle.backward(input, gradOutput)
    }
    val output = Tools.dense(graphModel.forward(input)).toTensor[Float]
    val outputDnn = Tools.dense(dnnModle.forward(input)).toTensor[Float]

    val gradInput = Tools.dense(graphModel.backward(input, gradOutput)).toTensor[Float]
    val gradInputDnn = Tools.dense(dnnModle.backward(input, gradOutput)).toTensor[Float]

    output.almostEqual(outputDnn, 1e-4) should be(true)
    gradInput.almostEqual(gradInputDnn, 1e-4) should be (true)

    val p1 = dnnModle.getParameters()
    val p2 = graphModel.getParameters()
    p1._1.almostEqual(p2._1, 1e-4) should be(true)
    p1._2 almostEqual(p2._2, 1e-4) should be(true)
  }

  "Dnn Lenet graph model" should "be correct" in {
    val batchSize = 2
    val seed = 1
    val inputFormat = Memory.Format.nchw
    val inputShape = Array(batchSize, 1, 28, 28)

    RNG.setSeed(seed)
    val graphModel = LeNet5.dnnGraph(batchSize, 10)
    RNG.setSeed(seed)
    val dnnModle = LeNet5.dnn(batchSize, 10)

    // graphModel.asInstanceOf[DnnGraph].compile(TrainingPhase)
    dnnModle.compile(TrainingPhase)

    val input = Tensor[Float](inputShape).rand()
    val gradOutput = Tensor[Float](batchSize, 10).rand()

    for (i <- 0 to 2) {
      graphModel.forward(input)
      dnnModle.forward(input)

      graphModel.backward(input, gradOutput)
      dnnModle.backward(input, gradOutput)
    }
    val output = Tools.dense(graphModel.forward(input)).toTensor[Float]
    val outputDnn = Tools.dense(dnnModle.forward(input)).toTensor[Float]

    val gradInput = Tools.dense(graphModel.backward(input, gradOutput)).toTensor[Float]
    val gradInputDnn = Tools.dense(dnnModle.backward(input, gradOutput)).toTensor[Float]

    output.almostEqual(outputDnn, 1e-4) should be(true)
    gradInput.almostEqual(gradInputDnn, 1e-4) should be (true)

    val p1 = dnnModle.getParameters()
    val p2 = graphModel.getParameters()
    p1._1.almostEqual(p2._1, 1e-4) should be(true)
    p1._2 almostEqual(p2._2, 1e-4) should be(true)
  }

  "ResNet50 graph model" should "be correct" in {
    val batchSize = 2
    val seed = 1
    val inputFormat = Memory.Format.nchw
    val inputShape = Array(batchSize, 3, 224, 224)

    RNG.setSeed(seed)
    val dnnModle = mkldnn.ResNet(batchSize, 1000, T("depth" -> 50,
      "dataSet" -> ResNet.DatasetType.ImageNet))
    RNG.setSeed(seed)
    val graphModel = mkldnn.ResNet.graph(batchSize, 1000, T("depth" -> 50,
      "dataSet" -> ResNet.DatasetType.ImageNet))

    val input = Tensor[Float](inputShape).rand()
    val gradOutput = Tensor[Float](batchSize, 1000).rand()

    // graphModel.asInstanceOf[DnnGraph].compile(TrainingPhase)
    dnnModle.compile(TrainingPhase)

    for (i <- 0 to 2) {
      graphModel.forward(input)
      dnnModle.forward(input)

      graphModel.backward(input, gradOutput)
      dnnModle.backward(input, gradOutput)
    }
    val output = Tools.dense(graphModel.forward(input)).toTensor[Float]
    val outputDnn = Tools.dense(dnnModle.forward(input)).toTensor[Float]

    val gradInput = Tools.dense(graphModel.backward(input, gradOutput)).toTensor[Float]
    val gradInputDnn = Tools.dense(dnnModle.backward(input, gradOutput)).toTensor[Float]

    output.almostEqual(outputDnn, 1e-4) should be(true)
    gradInput.almostEqual(gradInputDnn, 1e-4) should be (true)

    val p1 = graphModel.getParametersTable()
    val p2 = dnnModle.getParametersTable()
    val keys = p1.keySet
    for (i <- keys) {
      val k = i.asInstanceOf[String]
      val t1 = p1[Table](k)
      val t2 = p2[Table](k)
      t1 should be(t2)
    }
  }
}