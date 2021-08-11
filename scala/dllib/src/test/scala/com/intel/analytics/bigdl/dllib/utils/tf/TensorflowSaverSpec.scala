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
package com.intel.analytics.bigdl.utils.tf


import java.nio.ByteOrder
import java.util.UUID

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, DataFormat}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{T, Table}
import org.apache.log4j.Logger

class TensorflowSaverSpec extends TensorflowSpecHelper {
  override def doBefore(): Unit = {
    System.setProperty("bigdl.enableNHWC", "true")
  }

  override def doAfter(): Unit = {
    System.setProperty("bigdl.enableNHWC", "false")
  }

  "ReLU layer" should "be correctly saved" in {
    val inputTensor = Tensor[Float](T(
      T(1.0f, 2.0f, 5.0f, 6.0f),
      T(-3.0f, -4.0f, -7.0f, -8.0f)
    ))
    test(ReLU[Float](), inputTensor) should be(true)
  }

  "Linear layer" should "be correctly saved" in {
    val layer = Linear[Float](3, 4,
      initWeight = Tensor(T(
        T(1.0f, 2.0f, 3.0f),
        T(4.0f, 5.0f, 6.0f),
        T(7.0f, 8.0f, 9.0f),
        T(10.0f, 11.0f, 12.0f)
      )),
      initBias = Tensor(T(1.0f, 2.0f, 3.0f, 4.0f))
    )
    val input = Tensor[Float](T(
      T(1.0f, 2.0f, 5.0f),
      T(-3.0f, -4.0f, -7.0f)
    ))
    test(layer, input, "/add") should be(true)
  }

  "AvgPooling NHWC" should "be correctly saved" in {
    val layer = SpatialAveragePooling(2, 2, format = DataFormat.NHWC)
    val input = Tensor[Float](T(T(
      T(
        T(1.0f, 2.0f, 5.0f),
        T(-3.0f, -4.0f, -7.0f),
        T(-4.0f, -2.0f, -1.0f)
      ),
      T(
        T(-1.0f, -2.0f, -5.0f),
        T(3.0f, 4.0f, 7.0f),
        T(4.0f, 2.0f, 1.0f)
      )
    )))
    test(layer, input) should be(true)
  }

  "MaxPooling NHWC" should "be correctly saved" in {
    val layer = SpatialMaxPooling(2, 2, format = DataFormat.NHWC)
    val input = Tensor[Float](T(T(
      T(
        T(1.0f, 2.0f, 5.0f),
        T(-3.0f, -4.0f, -7.0f),
        T(-4.0f, -2.0f, -1.0f)
      ),
      T(
        T(-1.0f, -2.0f, -5.0f),
        T(3.0f, 4.0f, 7.0f),
        T(4.0f, 2.0f, 1.0f)
      )
    )))
    test(layer, input) should be(true)
  }

  "Tanh" should "be correctly saved" in {
    val layer = Tanh()
    val input = Tensor[Float](4).rand()
    test(layer, input) should be(true)
  }

  "Squeeze" should "be correctly saved" in {
    System.setProperty("bigdl.enableNHWC", "false")
    val layer = Squeeze(3).asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]]
    val input = Tensor[Float](4, 2, 1, 2).rand()
    test(layer, input) should be(true)
  }

  "CAddTableToTF" should "be correct" in {
    val layer = CAddTable[Float]()
    val input1 = Tensor[Float](4, 2, 2).rand()
    val input2 = Tensor[Float](4, 2, 2).rand()
    testMultiInput(layer, Seq(input1, input2), false) should be(true)
  }

  "CMultToTF" should "be correct" in {
    val layer = CMulTable[Float]()
    val input1 = Tensor[Float](4, 2, 2).rand()
    val input2 = Tensor[Float](4, 2, 2).rand()
    testMultiInput(layer, Seq(input1, input2), false) should be(true)
  }

  "JoinTableToTF" should "be correct" in {
    val layer = JoinTable[Float](3, -1)
    val input1 = Tensor[Float](4, 2, 2).rand()
    val input2 = Tensor[Float](4, 2, 2).rand()
    testMultiInput(layer.asInstanceOf[AbstractModule[Table, Tensor[Float], Float]],
      Seq(input1, input2), false) should be(true)
  }

  "LogSoftMax" should "be correctly saved" in {
    val layer = LogSoftMax()
    val input = Tensor[Float](4, 5).rand()
    test(layer, input) should be(true)
  }

  "SoftMax" should "be correctly saved" in {
    val layer = SoftMax()
    val input = Tensor[Float](4, 5).rand()
    test(layer, input) should be(true)
  }

  "Sigmoid" should "be correctly saved" in {
    val layer = Sigmoid()
    val input = Tensor[Float](4, 5).rand()
    test(layer, input) should be(true)
  }

  "SpatialConvolution NHWC" should "be correctly saved" in {
    val layer = SpatialConvolution(3, 5, 2, 2, format = DataFormat.NHWC)
    val input = Tensor[Float](4, 5, 5, 3).rand()
    test(layer, input, "/biasAdd") should be(true)
  }

  "SpatialConvolution NHWC without bias" should "be correctly saved" in {
    val layer = SpatialConvolution(3, 5, 2, 2, format = DataFormat.NHWC, withBias = false)
    val input = Tensor[Float](4, 5, 5, 3).rand()
    test(layer, input, "/conv2D") should be(true)
  }

  "SpatialConvolution NCHW with conv group" should "be correctly saved" in {
    cancel("tf cpu only support NHWC, this can be test on tf with MKL")
    val layer = SpatialConvolution(6, 10, 2, 2, nGroup = 2)
    val input = Tensor[Float](4, 6, 24, 24).rand()
    test(layer, input, "/concat/output") should be(true)
  }

  "SpatialConvolution NCHW with conv group without bias" should "be correctly saved" in {
    cancel("tf cpu only support NHWC, this can be test on tf with MKL")
    val layer = SpatialConvolution(6, 10, 2, 2, nGroup = 2, withBias = false)
    val input = Tensor[Float](4, 6, 24, 24).rand()
    test(layer, input, "/concat/output") should be(true)
  }

  "Scale" should "be correctly saved" in {
    val layer = Scale[Float](Array(10))
    val input = Tensor[Float](4, 10)
    test(layer, input, "/add/add") should be(true)
  }

  "TemporalConvolution" should "be correctly saved" in {
    val layer = TemporalConvolution(3, 5, 2, 2)
    val input = Tensor[Float](4, 16, 3).rand()
    test(layer, input, "/biasAdd") should be(true)
  }

  "Mean" should "be correctly saved" in {
    val layer = Mean(1, -1, true)
    val input = Tensor[Float](4, 5).rand()
    test(layer, input, "/output") should be(true)
  }

  "Padding" should "be correctly saved" in {
    val layer = Padding(1, 2, 2)
    val input = Tensor[Float](4, 5).rand()
    test(layer, input, "/output") should be(true)
  }

  "Batch Norm2D NCHW" should "be correctly saved" in {
    val layer = SpatialBatchNormalization(2)
    layer.evaluate()
    layer.weight.rand(10.0, 20.0)
    layer.bias.rand()
    layer.runningVar.rand(0.9, 1.1)
    layer.runningMean.rand()
    val input = Tensor[Float](3, 2, 4, 5).rand()
    test(layer, input, "/output") should be(true)
  }

  "Batch Norm2D NCHW without affine" should "be correctly saved" in {
    val layer = SpatialBatchNormalization(2, affine = false)
    layer.evaluate()
    layer.runningVar.resize(2).rand(0.9, 1.1)
    layer.runningMean.resize(2).rand()
    val input = Tensor[Float](3, 2, 4, 5).rand()
    test(layer, input, "/output") should be(true)
  }

  "Dropout" should "be correctly saved" in {
    val layer = Dropout()
    layer.evaluate()
    val input = Tensor[Float](3, 2).rand()
    test(layer, input) should be(true)
  }

  "View" should "be correctly saved when batch is enabled" in {
    val layer = View(2, 4)
    val input = Tensor[Float](4, 2, 2, 2).rand()
    test(layer, input) should be(true)
  }

  "LRN" should "be correct in NHWC" in {
    val layer = SpatialCrossMapLRN(format = DataFormat.NHWC)
    val input = Tensor[Float](4, 24, 24, 3).rand()
    test(layer, input) should be(true)
  }

  "LRN" should "be correct in NCHW" in {
    val layer = SpatialCrossMapLRN(format = DataFormat.NCHW)
    val input = Tensor[Float](4, 3, 24, 24).rand()
    test(layer, input, "/transpose2") should be(true)
  }

  "Reshape" should "be correctly saved" in {
    val layer = Reshape(Array(2, 4))
    val input = Tensor[Float](2, 2, 2).rand()
    test(layer, input) should be(true)
  }

  "lenet NHWC" should "be correctly saved" in {
    tfCheck()
    val conv1 = SpatialConvolution(1, 6, 5, 5, format = DataFormat.NHWC)
      .setName("conv1").inputs()
    val tanh1 = Tanh().setName("tanh1").inputs(conv1)
    val pool1 = SpatialMaxPooling(2, 2, 2, 2, format = DataFormat.NHWC)
      .setName("pool1").inputs(tanh1)
    val tanh2 = Tanh().setName("tanh2").inputs(pool1)
    val conv2 = SpatialConvolution(6, 12, 5, 5, format = DataFormat.NHWC)
      .setName("conv2").inputs(tanh2)
    val pool2 = SpatialMaxPooling(2, 2, 2, 2, format = DataFormat.NHWC)
      .setName("output").inputs(conv2)

    val funcModel = Graph(conv1, pool2)
    val inputData = Tensor(4, 28, 28, 1).rand()
    val outputData = funcModel.forward(inputData).toTensor

    val tmpFile = java.io.File.createTempFile("tensorflowSaverTest" + UUID.randomUUID(), "lenet")
    TensorflowSaver.saveGraphWithNodeDef(
      funcModel,
      Seq(Tensorflow.const(inputData, "input", ByteOrder.LITTLE_ENDIAN)),
      tmpFile.getPath,
      ByteOrder.LITTLE_ENDIAN,
      Set(Tensorflow.const(outputData, "target", ByteOrder.LITTLE_ENDIAN))
    )

    runPythonSaveTest(tmpFile.getPath, "") should be(true)
  }

  private def test(layer: AbstractModule[Tensor[Float], Tensor[Float], Float],
                   inputTensor: Tensor[Float],
                   outputSuffix: String = "") : Boolean = {
    tfCheck()
    val layerNode = layer.setName("output").inputs()
    val graph = Graph(layerNode, layerNode)
    val outputTensor = layer.forward(inputTensor)

    val tmpFile = java.io.File.createTempFile("tensorflowSaverTest" + UUID.randomUUID(), "Layer")
    logger.info(s"Save model to ${tmpFile}")

    TensorflowSaver.saveGraphWithNodeDef(
      graph,
      Seq(Tensorflow.const(inputTensor, "input", ByteOrder.LITTLE_ENDIAN)),
      tmpFile.getPath,
      ByteOrder.LITTLE_ENDIAN,
      Set(Tensorflow.const(outputTensor, "target", ByteOrder.LITTLE_ENDIAN))
    )
    runPythonSaveTest(tmpFile.getPath, outputSuffix)
  }

  private def testMultiInput(layer: AbstractModule[Table, Tensor[Float], Float],
                   inputTensors: Seq[Tensor[Float]],
                   convertNHWC: Boolean = false,
                   outputSuffix: String = "") : Boolean = {
    tfCheck()
    val layerNode = layer.setName("output").inputs()
    val inputNodes = inputTensors.map(_ => Input[Float]()).toArray
    inputNodes.foreach(_ -> layerNode)
    inputNodes.zipWithIndex.foreach(n => n._1.element.setName("inputNode" + n._2))
    val graph = Graph(inputNodes, layerNode)
    val inputTable = T()
    inputTensors.foreach(inputTable.insert(_))
    val outputTensor = layer.forward(inputTable)

    val tmpFile = java.io.File.createTempFile("tensorflowSaverTest" + UUID.randomUUID(), "Layer")
    logger.info(s"Save model to ${tmpFile}")
    val tfTensors = if (convertNHWC) {
      inputTensors.map(t => t.transpose(2, 3).transpose(3, 4).contiguous())
    } else {
      inputTensors
    }
    val outputSave = if (convertNHWC) {
      outputTensor.transpose(2, 3).transpose(3, 4).contiguous()
    } else {
      outputTensor
    }

    TensorflowSaver.saveGraphWithNodeDef(
      graph,
      tfTensors.zipWithIndex.map(t =>
        Tensorflow.const(t._1, "input" + t._2, ByteOrder.LITTLE_ENDIAN)),
      tmpFile.getPath,
      ByteOrder.LITTLE_ENDIAN,
      Set(Tensorflow.const(outputSave, "target", ByteOrder.LITTLE_ENDIAN))
    )
    runPythonSaveTest(tmpFile.getPath, outputSuffix)
  }
}
