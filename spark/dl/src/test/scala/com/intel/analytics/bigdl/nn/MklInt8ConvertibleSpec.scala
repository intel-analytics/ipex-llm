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

import java.io.File
import java.util.UUID

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.mkldnn.DnnGraph
import com.intel.analytics.bigdl.nn.mkldnn.Phase.InferencePhase
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}


class MklInt8ConvertibleSpec extends FlatSpec with Matchers with BeforeAndAfter  {

  val modelPath: String = "myTestModel" + UUID.randomUUID().toString
  val weightPath: String = "myTestModelWeight" + UUID.randomUUID().toString

  "Unit test setInputDimMask" should "work properly" in {
    val conv1 = SpatialConvolution(1, 6, 5, 5).setName("conv1")
    val conv2 = SpatialConvolution(1, 6, 5, 5).setName("conv2")
    val seq = Sequential().add(conv1).add(conv2)

    // Input mask default to 0
    seq.getInputDimMask() should be (0)
    conv1.getInputDimMask() should be (0)
    conv2.getInputDimMask() should be (0)

    // Sequential sets input dimension mask to 1 with recursive flag off
    // the submodules conv1 & conv2 should not be effected
    seq.setInputDimMask(1, false)
    seq.getInputDimMask() should be (1)
    conv1.getInputDimMask() should be (0)
    conv2.getInputDimMask() should be (0)

    // Sequential sets input dimension mask to 1 with recursive flag on
    // the submodules conv1 & conv2 should be effected
    seq.setInputDimMask(2, true)
    seq.getInputDimMask() should be (2)
    conv1.getInputDimMask() should be (2)
    conv2.getInputDimMask() should be (2)

    // change conv1's input dimension mask
    conv1.setInputDimMask(4, false)
    seq.getInputDimMask() should be (2)
    conv1.getInputDimMask() should be (4)
    conv2.getInputDimMask() should be (2)

    // change conv2's input dimension mask
    conv2.setInputDimMask(8, false)
    seq.getInputDimMask() should be (2)
    conv1.getInputDimMask() should be (4)
    conv2.getInputDimMask() should be (8)

  }


  "Unit test setOutputDimMask" should "work properly" in {
    val conv1 = SpatialConvolution(1, 6, 5, 5).setName("conv1")
    val conv2 = SpatialConvolution(1, 6, 5, 5).setName("conv2")
    val seq = Sequential().add(conv1).add(conv2)

    // Input mask default to 0
    seq.getOutputDimMask() should be (0)
    conv1.getOutputDimMask() should be (0)
    conv2.getOutputDimMask() should be (0)

    // Sequential sets input dimension mask to 1 with recursive flag off
    // the submodules conv1 & conv2 should not be effected
    seq.setOutputDimMask(1, false)
    seq.getOutputDimMask() should be (1)
    conv1.getOutputDimMask() should be (0)
    conv2.getOutputDimMask() should be (0)

    // Sequential sets input dimension mask to 1 with recursive flag on
    // the submodules conv1 & conv2 should be effected
    seq.setOutputDimMask(2, true)
    seq.getOutputDimMask() should be (2)
    conv1.getOutputDimMask() should be (2)
    conv2.getOutputDimMask() should be (2)

    // change conv1's input dimension mask
    conv1.setOutputDimMask(4, false)
    seq.getOutputDimMask() should be (2)
    conv1.getOutputDimMask() should be (4)
    conv2.getOutputDimMask() should be (2)

    // change conv2's input dimension mask
    conv2.setOutputDimMask(8, false)
    seq.getOutputDimMask() should be (2)
    conv1.getOutputDimMask() should be (4)
    conv2.getOutputDimMask() should be (8)

  }

  "Unit test setWeightDimMask" should "work properly" in {
    val conv1 = SpatialConvolution(1, 6, 5, 5).setName("conv1")
    val conv2 = SpatialConvolution(1, 6, 5, 5).setName("conv2")
    val seq = Sequential().add(conv1).add(conv2)

    // Input mask default to 0
    seq.getWeightDimMask() should be (0)
    conv1.getWeightDimMask() should be (0)
    conv2.getWeightDimMask() should be (0)

    // Sequential sets input dimension mask to 1 with recursive flag off
    // the submodules conv1 & conv2 should not be effected
    seq.setWeightDimMask(1, false)
    seq.getWeightDimMask() should be (1)
    conv1.getWeightDimMask() should be (0)
    conv2.getWeightDimMask() should be (0)

    // Sequential sets input dimension mask to 1 with recursive flag on
    // the submodules conv1 & conv2 should be effected
    seq.setWeightDimMask(2, true)
    seq.getWeightDimMask() should be (2)
    conv1.getWeightDimMask() should be (2)
    conv2.getWeightDimMask() should be (2)

    // change conv1's input dimension mask
    conv1.setWeightDimMask(4, false)
    seq.getWeightDimMask() should be (2)
    conv1.getWeightDimMask() should be (4)
    conv2.getWeightDimMask() should be (2)

    // change conv2's input dimension mask
    conv2.setWeightDimMask(8, false)
    seq.getWeightDimMask() should be (2)
    conv1.getWeightDimMask() should be (4)
    conv2.getWeightDimMask() should be (8)

  }

  "Calculating scales" should "work correct for BLAS Linear Module" in {

    val sampleMax = 999
    val inputSize = 120
    val outputSize = 1
    var inputMask = 0
    var outputMask = 0
    val inputTensor = make1DTensor(inputSize, sampleMax)

    // Global mask, null input
    val linear0 = Linear[Float](inputSize, outputSize)
    linear0.calcScales(null)
    linear0.output.isEmpty should be (true)
    linear0.getInputScales().isEmpty should be (true)
    linear0.getOutputScales().isEmpty should be (true)
    linear0.getWeightScales().isEmpty should be (true)

    // Global mask, non-null input
    val linear1 = Linear[Float](inputSize, outputSize)
    linear1.forward(inputTensor)
    linear1.calcScales(inputTensor)
    linear1.getInputScales() should be (Array(Array[Float](sampleMax)))
    linear1.getOutputScales().length should be (1)
    linear1.getOutputScales()(0).length should be (1)
    linear1.getWeightScales().length should be (1)
    linear1.getWeightScales()(0).length should be (1)

    // Single dimension mask, non-null input
    val linear2 = Linear[Float](inputSize, outputSize)
    inputMask = Math.pow(2, 0).toInt
    outputMask = Math.pow(2, 0).toInt
    linear2.setInputDimMask(inputMask, true)
    linear2.setOutputDimMask(outputMask, true)

    linear2.forward(inputTensor)
    linear2.calcScales(inputTensor)
    val output2 = linear2.output
    linear2.getInputScales() should be (Array(getScalesFromTensor(inputTensor, inputMask)))
    linear2.getOutputScales() should be (Array(getScalesFromTensor(output2, outputMask)))

    linear2.saveModule(modelPath, weightPath, true)

    val loadedModule2 = Module.loadModule[Float](modelPath, weightPath)
      .asInstanceOf[MklInt8Convertible]
    compareModules(linear2, loadedModule2)
  }

  "Calculating scales" should "work correct for DNN Linear Module" in {
    import com.intel.analytics.bigdl.mkl.Memory

    val sampleMax = 999
    val inputSize = 2
    val outputSize = 2
    var inputMask = 0
    var outputMask = 0
    val inputTensor = Tensor[Float](Array(4, inputSize)).rand(-1, 1)

    // Global mask, null input
    val linear0 = mkldnn.Linear(inputSize, outputSize)
    linear0.calcScales(null)

    linear0.getInputScales().isEmpty should be (true)
    linear0.getOutputScales().isEmpty should be (true)
    linear0.getWeightScales().isEmpty should be (true)

    // Global mask, non-null input
    val linear1 = mkldnn.Linear(inputSize, outputSize)
    val seq1 = mkldnn.Sequential()
      .add(mkldnn.Input(Array(4, inputSize), Memory.Format.nc))
      .add(linear1)
      .add(mkldnn.Output(Memory.Format.nc))

    seq1.compile(InferencePhase)
    seq1.forward(inputTensor)
    seq1.calcScales(inputTensor)
    linear1.getInputScales() should be (Array(Array[Float](inputTensor.abs().max())))
    linear1.getOutputScales().length should be (1)
    linear1.getOutputScales()(0).length should be (1)
    linear1.getWeightScales().length should be (1)
    linear1.getWeightScales()(0).length should be (1)

    // Single dimension mask, non-null input
    val linear2 = mkldnn.Linear(inputSize, outputSize)
    val seq2 = mkldnn.Sequential()
      .add(mkldnn.Input(Array(4, inputSize), Memory.Format.nc))
      .add(linear2)
      .add(mkldnn.Output(Memory.Format.nc))
    seq2.compile(InferencePhase)

    inputMask = Math.pow(2, 0).toInt
    outputMask = Math.pow(2, 0).toInt
    linear2.setInputDimMask(inputMask, true)
    linear2.setOutputDimMask(outputMask, true)

    seq2.forward(inputTensor)
    seq2.calcScales(inputTensor)

    val output2 = seq2.output.toTensor[Float]
    linear2.getInputScales() should be (Array(getScalesFromTensor(inputTensor, inputMask)))
    linear2.getOutputScales() should be (Array(getScalesFromTensor(output2, outputMask)))

    // for dnn linear, we skip the saveModule, because we do not support
  }

  private def compareModules(modX: MklInt8Convertible, modY: MklInt8Convertible): Unit = {
    modX.getInputDimMask() should be (modY.getInputDimMask())
    modX.getOutputDimMask() should be (modY.getOutputDimMask())
    modX.getWeightDimMask() should be (modY.getWeightDimMask())
    modX.getInputScales() should be (modY.getInputScales())
    modX.getOutputScales() should be (modY.getOutputScales())
    modX.getWeightScales() should be (modY.getWeightScales())
  }


  "Calculating scales" should "work correct for BLAS Spatial Convolution Module" in {
    val inputSize = 1
    val outputSize = 1
    val sampleMax = 999
    var dimMaskIdx = 0
    val inputTensor = make2DTensor().reshape(Array(inputSize, 3, 4))

    // Global mask, null input
    val spatialConv0 = SpatialConvolution[Float](inputSize, outputSize, 1, 1)
    spatialConv0.calcScales(null)
    spatialConv0.output.isEmpty should be (true)
    spatialConv0.getInputScales().isEmpty should be (true)
    spatialConv0.getOutputScales().isEmpty should be (true)
    spatialConv0.getWeightScales().isEmpty should be (true)

    // Global mask, non-null input
    val spatialConv1 = SpatialConvolution[Float](inputSize, outputSize, 1, 1)
    spatialConv1.forward(inputTensor)
    spatialConv1.calcScales(inputTensor)
    spatialConv1.getInputScales() should be (Array(Array[Float](12)))
    spatialConv1.getOutputScales().length should be (1)
    spatialConv1.getOutputScales()(0).length should be (1)
    spatialConv1.getWeightScales().length should be (1)
    spatialConv1.getWeightScales()(0).length should be (1)

    // Single input dimension mask, non-null input
    dimMaskIdx = 1
    val spatialConv2 = SpatialConvolution[Float](inputSize, outputSize, 1, 1)
    spatialConv2.setInputDimMask(Math.pow(2, dimMaskIdx - 1).toInt, true)
    spatialConv2.forward(inputTensor)
    spatialConv2.calcScales(inputTensor)
    val inputScales2 = Array(Array(inputTensor.select(dimMaskIdx, 1).max()))
    spatialConv2.getInputScales() should be (inputScales2)

    dimMaskIdx = 2
    val spatialConv3 = SpatialConvolution[Float](inputSize, outputSize, 1, 1)
    spatialConv3.setInputDimMask(Math.pow(2, dimMaskIdx - 1).toInt, true)
    spatialConv3.forward(inputTensor)
    spatialConv3.calcScales(inputTensor)
    val inputScales3 = Array((1 to inputTensor.size(dimMaskIdx)).map(
      idx => inputTensor.select(dimMaskIdx, idx).max()
    ).toArray)
    spatialConv3.getInputScales() should be (inputScales3)

    dimMaskIdx = 3
    val spatialConv4 = SpatialConvolution[Float](inputSize, outputSize, 1, 1)
    spatialConv4.setInputDimMask(Math.pow(2, dimMaskIdx - 1).toInt, true)
    spatialConv4.forward(inputTensor)
    spatialConv4.calcScales(inputTensor)
    val inputScales4 = Array((1 to inputTensor.size(dimMaskIdx)).map(
      idx => inputTensor.select(dimMaskIdx, idx).max()
    ).toArray)
    spatialConv4.getInputScales() should be (inputScales4)

    spatialConv4.saveModule(modelPath, weightPath, true)

    val loadedModule4 = Module.loadModule[Float](modelPath, weightPath)
      .asInstanceOf[MklInt8Convertible]
    compareModules(spatialConv4, loadedModule4)
  }

  "Calculating scales" should "work correct for DNN Spatial Convolution Module" in {
    import com.intel.analytics.bigdl.mkl.Memory
    val inputSize = 8
    val outputSize = 8
    var dimMaskIdx = 0
    val input = Tensor[Float](4, 8, 8, 8).rand(-1, 1)

    // Global mask, null input
    val spatialConv0 = mkldnn.SpatialConvolution(inputSize, outputSize, 1, 1)
    spatialConv0.calcScales(null)
    spatialConv0.getInputScales().isEmpty should be (true)
    spatialConv0.getOutputScales().isEmpty should be (true)
    spatialConv0.getWeightScales().isEmpty should be (true)

    // Global mask, non-null input
    val spatialConv1 = mkldnn.SpatialConvolution(inputSize, outputSize, 1, 1)
    val seq1 = mkldnn.Sequential()
      .add(mkldnn.Input(Array(4, 8, 8, 8), Memory.Format.nchw))
      .add(spatialConv1)
      .add(mkldnn.Output(Memory.Format.nchw))

    seq1.compile(InferencePhase)
    seq1.forward(input)
    spatialConv1.calcScales(input)

    spatialConv1.getInputScales() should be (Array(Array[Float](input.clone().abs().max())))
    spatialConv1.getOutputScales().length should be (1)
    spatialConv1.getOutputScales()(0).length should be (1)
    spatialConv1.getWeightScales().length should be (1)
    spatialConv1.getWeightScales()(0).length should be (1)

    seq1.release()

    // Single input dimension mask, non-null input
    dimMaskIdx = 1
    val spatialConv2 = mkldnn.SpatialConvolution(inputSize, outputSize, 1, 1)
    val seq2 = mkldnn.Sequential()
      .add(mkldnn.Input(Array(4, 8, 8, 8), Memory.Format.nchw))
      .add(spatialConv2)
      .add(mkldnn.Output(Memory.Format.nchw))
    seq2.compile(InferencePhase)
    seq2.forward(input)

    seq2.setInputDimMask(Math.pow(2, dimMaskIdx - 1).toInt, true)
    seq2.calcScales(input)

    spatialConv2.getInputScales().length should be (1)
    spatialConv2.getInputScales().flatten.length should be (4)

    seq2.release()

    dimMaskIdx = 2
    val spatialConv3 = mkldnn.SpatialConvolution(inputSize, outputSize, 1, 1)
    val seq3 = mkldnn.Sequential()
      .add(mkldnn.Input(Array(4, 8, 8, 8), Memory.Format.nchw))
      .add(spatialConv3)
      .add(mkldnn.Output(Memory.Format.nchw))
    seq3.compile(InferencePhase)
    seq3.forward(input)

    seq3.setInputDimMask(Math.pow(2, dimMaskIdx - 1).toInt, true)
    seq3.calcScales(input)

    val inputScales3 = Array((1 to input.size(dimMaskIdx)).map(
      idx => input.select(dimMaskIdx, idx).abs().max()
    ).toArray)
    spatialConv3.getInputScales() should be (inputScales3)

    seq3.release()

    dimMaskIdx = 3
    val spatialConv4 = mkldnn.SpatialConvolution(inputSize, outputSize, 1, 1)
    val seq4 = mkldnn.Sequential()
      .add(mkldnn.Input(Array(4, 8, 8, 8), Memory.Format.nchw))
      .add(spatialConv4)
      .add(mkldnn.Output(Memory.Format.nchw))
    seq4.compile(InferencePhase)
    seq4.forward(input)

    seq4.setInputDimMask(Math.pow(2, dimMaskIdx - 1).toInt, true)
    seq4.calcScales(input)
    val inputScales4 = Array((1 to input.size(dimMaskIdx)).map(
      idx => input.select(dimMaskIdx, idx).abs().max()
    ).toArray)
    spatialConv4.getInputScales() should be (inputScales4)
  }

  "Calculating scales" should "work correct for BLAS Sequential Module" in {
    var dimMaskIdx = 0
    var inputDimMask = 0
    var outputDimMask = 0

    def makeSequential(): Sequential[Float] = {
      val sequential = Sequential[Float]()
      sequential.add(Reshape[Float](Array(1, 28, 28)))
        .add(SpatialConvolution[Float](1, 6, 5, 5).setName("conv1_5x5"))
        .add(Tanh())
        .add(SpatialMaxPooling[Float](2, 2, 2, 2))
        .add(SpatialConvolution[Float](6, 12, 5, 5).setName("conv2_5x5"))
        .add(Tanh())
        .add(SpatialMaxPooling[Float](2, 2, 2, 2))
        .add(Reshape[Float](Array(12 * 4 * 4)))
        .add(Linear[Float](12 * 4 * 4, 100).setName("fc1"))
        .add(Tanh())
        .add(Linear[Float](100, 10).setName("fc2"))
        .add(LogSoftMax[Float]())
      sequential
    }

    val inputTensor = Tensor[Float](1, 28, 28).rand()

    // Global mask, null input
    val sequential0 = makeSequential()
    sequential0.calcScales(null)
    sequential0.output should be (null)
    sequential0.getInputScales().isEmpty should be (true)
    sequential0.getOutputScales().isEmpty should be (true)
    sequential0.getWeightScales().isEmpty should be (true)

    // Global mask, non-null input
    val sequential1 = makeSequential()
    sequential1.forward(inputTensor)
    sequential1.calcScales(inputTensor)
    sequential1.getInputScales().isEmpty should be (false)
    sequential1.getInputScales().length should be (1)
    sequential1.getInputScales()(0).length should be (1)
    sequential1.getOutputScales().isEmpty should be (false)
    sequential1.getOutputScales().length should be (1)
    sequential1.getOutputScales()(0).length should be (1)
    sequential1.getWeightScales().isEmpty should be (true)
    val inputScales1 = Array(Array(inputTensor.abs().max()))
    val outputScales1 = Array(Array(sequential1.output.toTensor[Float].abs().max()))
    sequential1.getInputScales() should be (inputScales1)
    sequential1.getOutputScales() should be (outputScales1)
    sequentialValidationHelper(sequential1)

    sequential1.saveModule(modelPath, weightPath, true)

    val loadedModule1 = Module.loadModule[Float](modelPath, weightPath)
      .asInstanceOf[MklInt8Convertible]
    compareModules(sequential1, loadedModule1)

    val sequential2 = makeSequential()
    sequential2.getInputDimMask() should be (0)
    sequential2.getOutputDimMask() should be (0)
    sequential2.getWeightDimMask() should be (0)
    sequential2.modules.filter(_.isInstanceOf[MklInt8Convertible]).foreach(x => {
      x.asInstanceOf[MklInt8Convertible].getInputDimMask() should be(0)
      x.asInstanceOf[MklInt8Convertible].getOutputDimMask() should be(0)
      x.asInstanceOf[MklInt8Convertible].getWeightDimMask() should be(0)
    })

    sequential2.setInputDimMask(2, true)
    sequential2.setOutputDimMask(2, true)
    sequential2.setWeightDimMask(2, true)

    sequential2.getInputDimMask() should be (2)
    sequential2.getOutputDimMask() should be (2)
    sequential2.getWeightDimMask() should be (2)
    sequential2.modules.filter(_.isInstanceOf[MklInt8Convertible]).foreach(x => {
      x.asInstanceOf[MklInt8Convertible].getInputDimMask() should be(2)
      x.asInstanceOf[MklInt8Convertible].getOutputDimMask() should be(2)
      x.asInstanceOf[MklInt8Convertible].getWeightDimMask() should be(2)
    })
  }


  "Calculating scales" should "work correct for BLAS ConcatTable Module" in {
    val sampleMax = 999
    val numElem = 12
    val inputTensor = make1DTensor(numElem, sampleMax)

    def makeConcatTable(): ConcatTable[Float] = {
      val concatTable = new  ConcatTable[Float]().setName("concatTable")
      concatTable.add(Linear[Float](numElem, 1).setName("A"))
      concatTable.add(Linear[Float](numElem, 1).setName("B"))
      concatTable
    }

    // Global mask, null input
    val concatTable0 = makeConcatTable()
    concatTable0.setInputDimMask(0, true)
    concatTable0.setOutputDimMask(0, true)
    concatTable0.setWeightDimMask(0, true)

    concatTable0.calcScales(null)
    concatTable0.getInputScales().isEmpty should be (true)
    concatTable0.getOutputScales().isEmpty should be (true)
    concatTable0.getWeightScales().isEmpty should be (true)

    // Global mask, non-null input
    val concatTable1 = makeConcatTable()

    concatTable1.forward(inputTensor)
    concatTable1.calcScales(inputTensor)
    concatTable1.getInputScales() should be (Array(Array[Float](sampleMax)))
    concatTable1.getOutputScales() should be (
      concatTable1.output.toTable.map((pair: (Any, Any)) => {
        val key = pair._1
        val value: Tensor[Float] = pair._2.asInstanceOf[Tensor[Float]]
        Array(value.abs().max())
      }).toArray
    )
    concatTableValidationHelper(inputTensor, concatTable1, 0)

    concatTable1.saveModule(modelPath, weightPath, true)

    val loadedModule1 = Module.loadModule[Float](modelPath, weightPath)
      .asInstanceOf[MklInt8Convertible]
    compareModules(concatTable1, loadedModule1)
  }

  "Calculating scales" should "work correct for BLAS CAddTable Module" in {
    val sampleMax = 999
    val numElem = 12
    val inputTable = T(
      Tensor[Float](Array(1.0f, 2.0f), Array(2)),
      Tensor[Float](Array(3.0f, 1.0f), Array(2)))

    val caddTable0 = CAddTable[Float]()
    caddTable0.setInputDimMask(0, true)
    caddTable0.setOutputDimMask(0, true)
    caddTable0.setWeightDimMask(0, true)

    caddTable0.calcScales(null)

    caddTable0.getInputScales().isEmpty should be (true)
    caddTable0.getOutputScales().isEmpty should be (true)
    caddTable0.getWeightScales().isEmpty should be (true)

    // Global mask, non-null input
    val caddTable1 = CAddTable()

    caddTable1.forward(inputTable)
    caddTable1.calcScales(inputTable)
    caddTable1.getOutputScales() should be (Array(Array[Float](4.0f)))
    caddTable1.getInputScales() should be (
      inputTable.toTable.map((pair: (Any, Any)) => {
        val key = pair._1
        val value: Tensor[Float] = pair._2.asInstanceOf[Tensor[Float]]
        Array(value.abs().max())
      }).toArray
    )

    caddTable1.saveModule(modelPath, weightPath, true)

    val loadedModule1 = Module.loadModule[Float](modelPath, weightPath)
      .asInstanceOf[MklInt8Convertible]
    compareModules(caddTable1, loadedModule1)
  }

  "Calculating scales" should "work correct for BLAS ReLU Module" in {
    val sampleMax = 999
    val numElem = 12
    val inputTensor = make1DTensor(numElem, sampleMax)

    val relu0 = ReLU[Float]()
    relu0.setInputDimMask(0, true)
    relu0.setOutputDimMask(0, true)
    relu0.setWeightDimMask(0, true)

    relu0.calcScales(null)

    relu0.getInputScales().isEmpty should be (true)
    relu0.getOutputScales().isEmpty should be (true)
    relu0.getWeightScales().isEmpty should be (true)

    // Global mask, non-null input
    val relu1 = ReLU[Float]()

    relu1.forward(inputTensor)
    relu1.calcScales(inputTensor)
    relu1.getInputScales() should be (Array(Array[Float](sampleMax)))
    relu1.getOutputScales() should be (Array(Array[Float](relu1.output.max())))

    relu1.saveModule(modelPath, weightPath, true)

    val loadedModule1 = Module.loadModule[Float](modelPath, weightPath)
      .asInstanceOf[MklInt8Convertible]
    compareModules(relu1, loadedModule1)
  }

  "Calculating scales" should "work correct for BLAS SpatialBatchNormalization Module" in {
    val numElem = 12
    val inputTensor = Tensor[Float](4, 2, 4, 4).rand(-100, 100)

    val bn0 = SpatialBatchNormalization[Float](2)
    bn0.setInputDimMask(0, true)
    bn0.setOutputDimMask(0, true)
    bn0.setWeightDimMask(0, true)

    bn0.calcScales(null)

    bn0.getInputScales().isEmpty should be (true)
    bn0.getOutputScales().isEmpty should be (true)
    bn0.getWeightScales().isEmpty should be (true)

    // Global mask, non-null input
    val bn1 = SpatialBatchNormalization[Float](2)

    bn1.forward(inputTensor)
    bn1.calcScales(inputTensor)
    bn1.getInputScales() should be (Array(Array[Float](inputTensor.abs().max())))
    bn1.getOutputScales() should be (Array(Array[Float](bn1.output.abs().max())))

    bn1.saveModule(modelPath, weightPath, true)

    val loadedModule1 = Module.loadModule[Float](modelPath, weightPath)
      .asInstanceOf[MklInt8Convertible]
    compareModules(bn1, loadedModule1)
  }

  "Calculating scales" should "work correct for Graph Module" in {
    def makeTestingGraph(): Graph[Float] = {
      val input = Reshape(Array(1, 28, 28)).inputs()
      val conv1 = SpatialConvolution(1, 6, 5, 5).setName("conv1_5x5").inputs(input)
      val tanh1 = Tanh().inputs(conv1)
      val pool1 = SpatialMaxPooling(2, 2, 2, 2).inputs(tanh1)
      val conv2 = SpatialConvolution(6, 12, 5, 5).setName("conv2_5x5").inputs(pool1)
      val tanh2 = Tanh().inputs(conv2)
      val pool2 = SpatialMaxPooling(2, 2, 2, 2).inputs(tanh2)
      val reshape = Reshape(Array(12 * 4 * 4)).inputs(pool2)
      val fc1 = Linear(12 * 4 * 4, 100).setName("fc1").inputs(reshape)
      val tanh3 = Tanh().inputs(fc1)
      val fc2 = Linear(100, 10).setName("fc2").inputs(tanh3)
      val output = LogSoftMax().inputs(fc2)

      Graph(input, output)
    }

    val inputTensor = Tensor(1, 28, 28).rand()

    // global mask, null input
    val graph0 = makeTestingGraph()
    graph0.setInputDimMask(0, true)
    graph0.setOutputDimMask(0, true)
    graph0.calcScales(null)
    graph0.getInputDimMask() should be (0)
    graph0.getOutputDimMask() should be (0)
    graph0.getInputScales().isEmpty should be (true)
    graph0.getOutputScales().isEmpty should be (true)

    // global mask, non-null input
    val graph1 = makeTestingGraph()
    graph1.setInputDimMask(0, true)
    graph1.setOutputDimMask(0, true)
    graph1.forward(inputTensor)
    graph1.calcScales(inputTensor)
    val graphOutput1 = graph1.output

    graph1.getInputDimMask() should be (0)
    graph1.getOutputDimMask() should be (0)
    graphOutput1 should not be (null)
    graph1.getInputScales() should be (Array(Array(inputTensor.abs().max())))
    graph1.getOutputScales() should be (Array(Array(graphOutput1.toTensor.abs().max())))
    graphValidationHelper(graph1, inputTensor)

    graph1.saveModule(modelPath, weightPath, true)

    val loadedGraph1 = Module.loadModule[Float](modelPath, weightPath)
      .asInstanceOf[MklInt8Convertible]
    compareModules(graph1, loadedGraph1)
  }

  "Calculating scales" should "work correct for DNN Graph Module" in {
    import com.intel.analytics.bigdl.mkl.Memory
    System.setProperty("bigdl.mkldnn.fusion", "false")

    def dnnGraph(batchSize: Int, classNum: Int): mkldnn.DnnGraph = {
      val inputShape = Array(batchSize, 1, 28, 28)
      val outputShape = Array(batchSize, 10)

      val input = mkldnn.Input(inputShape, Memory.Format.nchw).inputs()
      val conv1 = mkldnn.SpatialConvolution(1, 20, 5, 5).setName("conv1").inputs(input)
      val bn1 = mkldnn.SpatialBatchNormalization(20).setName("bn1").inputs(conv1)
      val pool1 = mkldnn.MaxPooling(2, 2, 2, 2).setName("pool1").inputs(bn1)
      val conv2 = mkldnn.SpatialConvolution(20, 50, 5, 5).setName("conv2").inputs(pool1)
      val pool2 = mkldnn.MaxPooling(2, 2, 2, 2).setName("pool2").inputs(conv2)
      val ip1 = mkldnn.Linear(50 * 4 * 4, 500).setName("ip1").inputs(pool2)
      val relu1 = mkldnn.ReLU().setName("relu1").inputs(ip1)
      val ip2 = mkldnn.Linear(500, 10).setName("ip2").inputs(relu1)
      val output = mkldnn.ReorderMemory(mkldnn.HeapData(outputShape, Memory.Format.nc)).inputs(ip2)

      val graph = DnnGraph(Array(input), Array(output))
      graph.evaluate()
      graph.compile(InferencePhase)
      graph
    }

    val inputTensor = Tensor(4, 1, 28, 28).rand()

    // global mask, null input
    val graph0 = dnnGraph(4, 10)
    graph0.setInputDimMask(0, true)
    graph0.setOutputDimMask(0, true)
    graph0.calcScales(null)
    graph0.getInputDimMask() should be (0)
    graph0.getOutputDimMask() should be (0)
    graph0.getInputScales().isEmpty should be (true)
    graph0.getOutputScales().isEmpty should be (true)
    graph0.release()

    // global mask, non-null input
    val graph1 = dnnGraph(4, 10)
    graph1.setInputDimMask(0, true)
    graph1.setOutputDimMask(0, true)
    graph1.setWeightDimMask(1, true)
    graph1.forward(inputTensor)
    graph1.calcScales(inputTensor)
    val graphOutput1 = graph1.output

    graph1.getInputDimMask() should be (0)
    graph1.getOutputDimMask() should be (0)
    graphOutput1 should not be (null)

    graph1.getForwardExecutions()
      .filter(_.element.isInstanceOf[mkldnn.SpatialConvolution])
      .map(_.element.asInstanceOf[mkldnn.SpatialConvolution])
      .map(x => x.nOutputPlane == x.getWeightScales().flatten.length)
      .exists(_ == false) should be (false)

    graph1.getForwardExecutions()
      .filter(_.element.isInstanceOf[mkldnn.SpatialBatchNormalization])
      .map(_.element.asInstanceOf[mkldnn.SpatialBatchNormalization])
      .map(x => x.getOutputScales().flatten.length == 1 && x.getInputScales().flatten.length == 1)
      .exists(_ == false) should be (false)

    graph1.getForwardExecutions()
      .filter(_.element.isInstanceOf[mkldnn.ReLU])
      .map(_.element.asInstanceOf[mkldnn.ReLU])
      .map(x => x.getOutputScales().flatten.length == 1 && x.getInputScales().flatten.length == 1)
      .exists(_ == false) should be (false)

    graph1.release()
    System.clearProperty("bigdl.mkldnn.fusion")
  }

  "calc scales with mask 3" should "work correctly" in {
    var i = 0f
    val tensor = Tensor(Array(4, 2, 2)).apply1(_ => {
      i = i + 1
      i
    })

    println(tensor)

    val mask0 = Utils.calcScales(tensor, 0)
    val mask1 = Utils.calcScales(tensor, 1)
    val mask2 = Utils.calcScales(tensor, 2)
    val mask3 = Utils.calcScales(tensor, 3)
    val mask4 = Utils.calcScales(tensor, 4)
    val mask5 = Utils.calcScales(tensor, 5)
    val mask6 = Utils.calcScales(tensor, 6)
    val mask7 = Utils.calcScales(tensor, 7)

    mask0 should be (Array(16.0))
    mask1 should be (Array(4, 8, 12, 16))
    mask2 should be (Array(14, 16))
    mask3 should be (Array(2, 4, 6, 8, 10, 12, 14, 16))
    mask4 should be (Array(15, 16))
    mask5 should be (Array(3, 4, 7, 8, 11, 12, 15, 16))
    mask6 should be (Array(13, 14, 15, 16))
    mask7 should be (Array(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16))
  }


  private def graphValidationHelper(graph: Graph[Float], inputActvt: Activity): Unit = {
    val nextNodes = graph.getForwardExecutions()
    var i = 0
    while (i < nextNodes.length) {
      val currNode = nextNodes(i)
      val currInputActvt = graph.findInput(currNode, inputActvt)
      val currOutputActvt = currNode.element.output
      if (currNode.element.isInstanceOf[MklInt8Convertible]) {
        val currNodeInt8 = currNode.element.asInstanceOf[MklInt8Convertible]
        val currInputScales = currNodeInt8.getInputScales()
        val currOutputScales = currNodeInt8.getOutputScales()
        currNodeInt8.getInputDimMask() should be (0)
        currNodeInt8.getOutputDimMask() should be (0)
        currNodeInt8.getInputScales() should be (Array(Array(currInputActvt.toTensor.abs().max())))
        currNodeInt8.getOutputScales() should be (
          Array(Array(currOutputActvt.toTensor.abs().max()))
        )
      }
      i += 1
    }
  }


  /**
   * Iterate over modules inside the Sequential module, verify their calculated scales
   * @param sequential the sequential to be verified
   */
  private def sequentialValidationHelper(sequential: Sequential[Float]): Unit = {

    var prevModule: AbstractModule[_, _, Float] = null
    val moduleIter = sequential.modules.iterator

    while (moduleIter.hasNext) {
      val currModule = moduleIter.next()
      if (currModule.isInstanceOf[MklInt8Convertible]) {
        val currInputMask = currModule.asInstanceOf[MklInt8Convertible].getInputDimMask()
        val currOutputMask = currModule.asInstanceOf[MklInt8Convertible].getOutputDimMask()
        val currInputScales = currModule.asInstanceOf[MklInt8Convertible].getInputScales()
        val currOutputScales = currModule.asInstanceOf[MklInt8Convertible].getOutputScales()
        if (prevModule != null) {
          val prevOutput = prevModule.output.asInstanceOf[Tensor[Float]]
          Array(getScalesFromTensor(prevOutput, currInputMask)) should be (currInputScales)
        }
        Array(getScalesFromTensor(currModule.output.toTensor[Float], currOutputMask)) should
          be (currOutputScales)
      }
      prevModule = currModule
    }
  }


  /**
   * Iterate over modules inside the ConcatTable module, verify their calculated scales
   * @param inputTensor input of the ConcatTable
   * @param concatTable the ConcatTable to be verified
   */
  private def concatTableValidationHelper(inputTensor: Tensor[Float],
    concatTable: ConcatTable[Float],
    mask: Int): Unit = {

    val moduleIter = concatTable.modules.iterator
    if (mask == 0) {
      while (moduleIter.hasNext) {
        val currModule = moduleIter.next()
        val currInputScales = currModule.asInstanceOf[MklInt8Convertible].getInputScales()
        val currOutputScales = currModule.asInstanceOf[MklInt8Convertible].getOutputScales()
        currModule.asInstanceOf[MklInt8Convertible].getInputDimMask() should be (0)
        currModule.asInstanceOf[MklInt8Convertible].getOutputDimMask() should be (0)
        inputTensor.max() should be (currInputScales(0)(0))
        currModule.output.toTensor[Float].max() should be (currOutputScales(0)(0))
      }
    } else {
      while (moduleIter.hasNext) {
        val currModule = moduleIter.next()
        val currInputScales = currModule.asInstanceOf[MklInt8Convertible].getInputScales()
        val currOutputScales = currModule.asInstanceOf[MklInt8Convertible].getOutputScales()
        val inputDimSize = inputTensor.size(mask)
        val outputDimSize = currModule.output.toTensor[Float].size(mask)

        (1 to inputDimSize).map(idx => {
          inputTensor.select(mask, idx).abs().max()
        }).toArray should be (currInputScales)

        (1 to outputDimSize).map(idx => {
          currModule.output.toTensor[Float].select(mask, idx).abs().max()
        }).toArray should be (currOutputScales)
      }

    }
  }


  /**
   * Calculate the scales based on the input tensor and dimension mask
   * @param tensor input tensor
   * @param mask dimension mask
   * @return an Array contains scales
   */
  private def getScalesFromTensor(tensor: Tensor[Float], mask: Int): Array[Float] = {

    if (mask == 0) {
      Array(tensor.abs().max())
    } else {
      val dimSize = tensor.size(mask)

      (1 to dimSize).map(idx => {
        tensor.select(mask, idx).abs().max()
      }).toArray
    }

  }


  /**
   * Helper method to make testing 2 dimensional tensor
   * tensor =
   * 01 10 03 12
   * 09 07 11 08
   * 05 02 06 04
   *
   * @return a 2D tensor of float
   */
  private def make2DTensor(): Tensor[Float] = {
    val tensor = Tensor[Float](3, 4)
    tensor.setValue(1, 1, 1)
    tensor.setValue(1, 2, 10)
    tensor.setValue(1, 3, 3)
    tensor.setValue(1, 4, 12)
    tensor.setValue(2, 1, 9)
    tensor.setValue(2, 2, 7)
    tensor.setValue(2, 3, 11)
    tensor.setValue(2, 4, 8)
    tensor.setValue(3, 1, 5)
    tensor.setValue(3, 2, 2)
    tensor.setValue(3, 3, 6)
    tensor.setValue(3, 4, 4)

    tensor
  }


  /**
   * Helper method to make testing 1 dimensional tensor
   * @param n tensor size
   * @param max max value of the random generated tensor
   * @return a tensor of float
   */
  private def make1DTensor(n: Int, max: Float): Tensor[Float] = {
    val tensor = Tensor[Float](n)
    tensor.rand(0, 100)
    tensor.setValue(1, max)
    tensor
  }


  after {
    new File(modelPath).delete()
    new File(weightPath).delete()
  }

}

