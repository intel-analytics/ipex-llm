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
import com.intel.analytics.bigdl.nn
import com.intel.analytics.bigdl.nn.StaticGraph
import com.intel.analytics.bigdl.nn.mkldnn.Phase.TrainingPhase
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.L2Regularizer
import com.intel.analytics.bigdl.tensor.{DnnStorage, Tensor}
import com.intel.analytics.bigdl.utils.{Engine, MklDnn, RandomGenerator}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.apache.commons.lang3.SerializationUtils
import org.scalatest.{FlatSpec, Matchers}

class LinearSpec extends FlatSpec with Matchers {

  "Convert blas linear to mkldnn with NHWC" should "be correct" in {
    Engine.setEngineType(MklDnn)
    RandomGenerator.RNG.setSeed(100)
    val linear = nn.Linear(12, 3)
    val linear2 = nn.Linear(12, 3)

    linear2.weight.copy(linear.weight)
    linear2.bias.copy(linear.bias)

    val blas = nn.Sequential()
    blas.add(nn.Reshape(Array(12)).setName("linear"))
    blas.add(linear)

    val blas2 = nn.Sequential().add(linear2).toGraph().
      asInstanceOf[StaticGraph[Float]]
    blas2.setInputFormats(Seq(Memory.Format.nhwc))
    blas2.setOutputFormats(Seq(Memory.Format.nc))
    val dnn = blas2.toIRgraph()

    val input = Tensor[Float](4, 2, 2, 3).rand() // nhwc

    val outBlas = blas.forward(input).toTensor[Float]
    val outDnn = dnn.forward(input).toTensor[Float]
    Equivalent.nearequals(outDnn.toTensor, outBlas.toTensor, 1e-5) should be (true)

    val gradOutput = Tensor[Float]().resizeAs(outBlas).copy(outBlas)
    val gradInputBlas = blas.backward(input, gradOutput)
    val gradInputDnn = dnn.backward(input, gradOutput)
    Equivalent.nearequals(gradInputBlas.toTensor, gradInputDnn.toTensor, 1e-5) should be (true)
  }

  "linear updateOutput" should "work correctly" in {
    val inputSize = 2
    val outputSize = 2
    val batchSize = 2

    val inputFormat = HeapData(Array(batchSize, inputSize), Memory.Format.nc)
    val outputFormat = HeapData(Array(batchSize, outputSize), Memory.Format.nc)
    val input = Tensor[Float](batchSize, inputSize).rand()

    val initWeight = Tensor[Float](outputSize, inputSize).rand()
    val initBias = Tensor[Float](outputSize).rand()

    val linear = Linear(inputSize, outputSize, initWeight = initWeight, initBias = initBias)
    linear.setRuntime(new MklDnnRuntime)
    linear.initFwdPrimitives(Array(inputFormat), TrainingPhase)
    linear.initBwdPrimitives(Array(outputFormat), TrainingPhase)
    linear.initGradWPrimitives(Array(outputFormat), TrainingPhase)

    val output = linear.forward(input)
    println(output)

    val nnLinear = nn.Linear(inputSize, outputSize, initWeight = initWeight, initBias = initBias)
    val nnOutput = nnLinear.forward(input)
    println(nnOutput)

    Tools.dense(output) should be (nnOutput)
  }

  "Dnn Linear with regularizer" should "work correctly" in {
    val inputSize = 2048
    val outputSize = 1000
    val batchSize = 2

    RNG.setSeed(1000)
    val inputFormat = HeapData(Array(batchSize, inputSize), Memory.Format.nc)
    val outputFormat = HeapData(Array(batchSize, outputSize), Memory.Format.nc)
    val input = Tensor[Float](batchSize, inputSize).rand()
    val gradOutput = Tensor[Float](batchSize, outputSize).rand()

    val initWeight = Tensor[Float](outputSize, inputSize).rand()
    val initBias = Tensor[Float](outputSize).rand()

    val wRegularizer = L2Regularizer(1e-4)
    val bRegularizer = L2Regularizer(1e-4)

    val linear = Linear(inputSize, outputSize, initWeight = initWeight, initBias = initBias,
      wRegularizer = wRegularizer, bRegularizer = bRegularizer)
    val nnLinear = nn.Linear(inputSize, outputSize, initWeight = initWeight, initBias = initBias,
      wRegularizer = wRegularizer, bRegularizer = bRegularizer)

    linear.setRuntime(new MklDnnRuntime)
    linear.initFwdPrimitives(Array(inputFormat), TrainingPhase)
    linear.initBwdPrimitives(Array(outputFormat), TrainingPhase)
    linear.initGradWPrimitives(Array(outputFormat), TrainingPhase)

    for (i <- 0 to 3) {
      linear.zeroGradParameters()
      nnLinear.zeroGradParameters()

      nnLinear.forward(input)
      nnLinear.backward(input, gradOutput)
      linear.forward(input)
      linear.backward(input, gradOutput)
    }

    Tools.dense(linear.output) should be (nnLinear.output)
    Tools.dense(linear.gradInput) should be (nnLinear.gradInput)

    val p1 = linear.getParameters()
    val p2 = linear.getParameters()

    p1._1 should be (p2._1)
    p1._2 should be (p2._2)
  }



  "linear updateOutput multi times" should "work correctly" in {
    val inputSize = 2
    val outputSize = 2
    val batchSize = 2

    val inputFormat = HeapData(Array(batchSize, inputSize), Memory.Format.nc)
    val outputFormat = HeapData(Array(batchSize, outputSize), Memory.Format.nc)

    val initWeight = Tensor[Float](outputSize, inputSize).rand()
    val initBias = Tensor[Float](outputSize).rand()

    val inputs = new Array[Tensor[Float]](100)
    for (i <- inputs.indices) {
      inputs(i) = Tensor[Float](batchSize, inputSize).rand()
    }

    val linear = Linear(inputSize, outputSize, initWeight = initWeight, initBias = initBias)
    linear.setRuntime(new MklDnnRuntime)
    linear.initFwdPrimitives(Array(inputFormat), TrainingPhase)
    linear.initBwdPrimitives(Array(outputFormat), TrainingPhase)
    linear.initGradWPrimitives(Array(outputFormat), TrainingPhase)

    for (in <- inputs) {
      linear.forward(in)
    }
    println(linear.output)

    val nnLinear = nn.Linear(inputSize, outputSize, initWeight = initWeight, initBias = initBias)
    for (in <- inputs) {
      nnLinear.forward(in)
    }
    println(nnLinear.output)

    Tools.dense(linear.output) should be (nnLinear.output)
  }

  "linear updateGradInput" should "work correctly" in {
    val inputSize = 2
    val outputSize = 2
    val batchSize = 2

    val inputFormat = HeapData(Array(batchSize, inputSize), Memory.Format.nc)
    val outputFormat = HeapData(Array(batchSize, outputSize), Memory.Format.nc)
    val input = Tensor[Float](batchSize, inputSize).rand()
    val gradOutput = Tensor().resize(outputFormat.shape).rand()

    val initWeight = Tensor[Float](outputSize, inputSize).rand()
    val initBias = Tensor[Float](outputSize).rand()

    val linear = Linear(inputSize, outputSize, initWeight = initWeight, initBias = initBias)
    linear.setRuntime(new MklDnnRuntime)
    linear.initFwdPrimitives(Array(inputFormat), TrainingPhase)
    linear.initBwdPrimitives(Array(outputFormat), TrainingPhase)
    linear.initGradWPrimitives(Array(outputFormat), TrainingPhase)

    val output = linear.forward(input)
    val gradInput = linear.updateGradInput(input, gradOutput)

    val nnLinear = nn.Linear(inputSize, outputSize, initWeight = initWeight, initBias = initBias)
    val nnOutput = nnLinear.forward(input)
    val nnGradInput = nnLinear.updateGradInput(input, gradOutput)

    println(gradInput)
    println("-" * 80)
    println(nnGradInput)

    Tools.dense(gradInput) should be (nnGradInput)
  }

  "linear updateGradInput multi times" should "work correctly" in {
    val inputSize = 2
    val outputSize = 2
    val batchSize = 2

    val inputFormat = HeapData(Array(batchSize, inputSize), Memory.Format.nc)
    val outputFormat = HeapData(Array(batchSize, outputSize), Memory.Format.nc)

    val initWeight = Tensor[Float](outputSize, inputSize).rand()
    val initBias = Tensor[Float](outputSize).rand()

    val inputs = new Array[Tensor[Float]](100)
    for (i <- inputs.indices) {
      inputs(i) = Tensor[Float](batchSize, inputSize).rand()
    }

    val linear = Linear(inputSize, outputSize, initWeight = initWeight, initBias = initBias)
    linear.setRuntime(new MklDnnRuntime)
    linear.initFwdPrimitives(Array(inputFormat), TrainingPhase)
    linear.initBwdPrimitives(Array(outputFormat), TrainingPhase)
    linear.initGradWPrimitives(Array(outputFormat), TrainingPhase)

    for (i <- inputs.indices) {
      inputs(i) = Tensor[Float](batchSize, inputSize).rand()
    }

    val gradOutputs = new Array[Tensor[Float]](100)
    for (i <- gradOutputs.indices) {
      gradOutputs(i) = Tensor[Float](batchSize, outputSize).rand()
    }

    linear.forward(inputs.last)

    for (i <- inputs.indices) {
      linear.updateGradInput(inputs(i), gradOutputs(i))
    }

    val nnLinear = nn.Linear(inputSize, outputSize, initWeight = initWeight, initBias = initBias)
    val nnOutput = nnLinear.forward(inputs.last)

    for (i <- inputs.indices) {
      nnLinear.updateGradInput(inputs(i), gradOutputs(i))
    }

    Tools.dense(linear.gradInput) should be (nnLinear.gradInput)
  }

  "linear accGradParameters" should "work correctly" in {
    val inputSize = 2
    val outputSize = 2
    val batchSize = 2

    val inputFormat = HeapData(Array(batchSize, inputSize), Memory.Format.nc)
    val outputFormat = HeapData(Array(batchSize, outputSize), Memory.Format.nc)
    val input = Tensor[Float](batchSize, inputSize).rand()
    val gradOutput = Tensor[Float]().resize(outputFormat.shape).rand()

    val initWeight = Tensor[Float](outputSize, inputSize).rand()
    val initBias = Tensor[Float](outputSize).rand()

    val linear = Linear(inputSize, outputSize, initWeight = initWeight, initBias = initBias)
    linear.setRuntime(new MklDnnRuntime)
    linear.initFwdPrimitives(Array(inputFormat), TrainingPhase)
    linear.initBwdPrimitives(Array(outputFormat), TrainingPhase)
    linear.initGradWPrimitives(Array(outputFormat), TrainingPhase)

    val output = linear.forward(input)

    val gradInput = linear.updateGradInput(input, gradOutput)

    val nnLinear = nn.Linear(inputSize, outputSize, initWeight = initWeight, initBias = initBias)
    val nnOutput = nnLinear.forward(input)
    val nnGradInput = nnLinear.updateGradInput(input, gradOutput)

    linear.accGradParameters(input, gradOutput)
    nnLinear.accGradParameters(input, gradOutput)

    println(linear.gradWeight)
    println(linear.gradBias)
    println("-" * 80)
    println(nnLinear.gradWeight)
    println(nnLinear.gradBias)

    Tools.dense(linear.gradWeight.native) should be (nnLinear.gradWeight)
    Tools.dense(linear.gradBias.native) should be (nnLinear.gradBias)
  }

  "linear cloned" should "work correctly" in {
    val inputSize = 2
    val outputSize = 2
    val batchSize = 2

    val inputFormat = HeapData(Array(batchSize, inputSize), Memory.Format.nc)
    val outputFormat = HeapData(Array(batchSize, outputSize), Memory.Format.nc)
    val input = Tensor[Float](batchSize, inputSize).rand()
    val gradOutput = Tensor[Float]().resize(outputFormat.shape).rand()

    val initWeight = Tensor[Float](outputSize, inputSize).rand()
    val initBias = Tensor[Float](outputSize).rand()

    val linear = Linear(inputSize, outputSize, initWeight = initWeight, initBias = initBias)
    linear.setRuntime(new MklDnnRuntime)
    linear.initFwdPrimitives(Array(inputFormat), TrainingPhase)
    linear.initBwdPrimitives(Array(outputFormat), TrainingPhase)
    linear.initGradWPrimitives(Array(outputFormat), TrainingPhase)

    val output = linear.forward(input)
    val gradInput = linear.updateGradInput(input, gradOutput)
    linear.accGradParameters(input, gradOutput)

    val cloned = SerializationUtils.clone(linear)
    cloned.setRuntime(new MklDnnRuntime)
    cloned.initFwdPrimitives(Array(inputFormat), TrainingPhase)
    cloned.initBwdPrimitives(Array(outputFormat), TrainingPhase)
    cloned.initGradWPrimitives(Array(outputFormat), TrainingPhase)

    cloned.forward(input)

    Tools.dense(linear.output) should be (Tools.dense(cloned.output))

    linear.backward(input, gradOutput)
    cloned.backward(input, gradOutput)

    Tools.dense(linear.gradInput) should be (Tools.dense(cloned.gradInput))
    Tools.dense(linear.gradWeight.native) should be (Tools.dense(cloned.gradWeight.native))
    Tools.dense(linear.gradBias.native) should be (Tools.dense(cloned.gradBias.native))
  }

  "linear released" should "work correctly" in {
    val inputSize = 2
    val outputSize = 2
    val batchSize = 2

    val inputFormat = HeapData(Array(batchSize, inputSize), Memory.Format.nc)
    val outputFormat = HeapData(Array(batchSize, outputSize), Memory.Format.nc)
    val input = Tensor[Float](batchSize, inputSize).rand()
    val gradOutput = Tensor[Float]().resize(outputFormat.shape).rand()

    val initWeight = Tensor[Float](outputSize, inputSize).rand()
    val initBias = Tensor[Float](outputSize).rand()

    val initCount = DnnStorage.get().count(!_._2)

    val linear = Linear(inputSize, outputSize, initWeight = initWeight, initBias = initBias)
    linear.setRuntime(new MklDnnRuntime)
    linear.initFwdPrimitives(Array(inputFormat), TrainingPhase)
    linear.initBwdPrimitives(Array(outputFormat), TrainingPhase)
    linear.initGradWPrimitives(Array(outputFormat), TrainingPhase)

    linear.forward(input)
    linear.backward(input, gradOutput)

    linear.release()
    DnnStorage.get().count(_._2 == false) should be (initCount)
  }

  "linear with dense weights" should "work correctly" in {
    val inputSize = 2
    val outputSize = 2
    val batchSize = 2

    val inputFormat = HeapData(Array(batchSize, inputSize), Memory.Format.nc)
    val outputFormat = HeapData(Array(batchSize, outputSize), Memory.Format.nc)
    val input = Tensor[Float](batchSize, inputSize).rand()
    val gradOutput = Tensor[Float]().resize(outputFormat.shape).rand()

    val initWeight1 = Tensor[Float](outputSize, inputSize).rand()
    val initBias1 = Tensor[Float](outputSize).rand()

    val initWeight2 = Tensor[Float](outputSize, inputSize).rand()
    val initBias2 = Tensor[Float](outputSize).rand()

    val linear1 = Linear(inputSize, outputSize, initWeight = initWeight1, initBias = initBias1)
    linear1.setRuntime(new MklDnnRuntime)
    linear1.initFwdPrimitives(Array(inputFormat), TrainingPhase)
    linear1.initBwdPrimitives(Array(outputFormat), TrainingPhase)
    linear1.initGradWPrimitives(Array(outputFormat), TrainingPhase)

    linear1.forward(input)
    linear1.backward(input, gradOutput)

    linear1.parameters()._1.zip(Array(initWeight2, initBias2)).foreach(x => x._1.copy(x._2))
    linear1.forward(input)
    linear1.backward(input, gradOutput)

    val linear2 = Linear(inputSize, outputSize, initWeight = initWeight2, initBias = initBias2)
    linear2.setRuntime(new MklDnnRuntime)
    linear2.initFwdPrimitives(Array(inputFormat), TrainingPhase)
    linear2.initBwdPrimitives(Array(outputFormat), TrainingPhase)
    linear2.initGradWPrimitives(Array(outputFormat), TrainingPhase)

    linear2.forward(input)
    linear2.backward(input, gradOutput)

    Tools.dense(linear1.output) should be (Tools.dense(linear2.output))
    Tools.dense(linear1.gradInput) should be (Tools.dense(linear2.gradInput))
    linear1.parameters()._2.zip(linear2.parameters()._2).foreach(x => x._1 should be (x._2))
  }

  "linear with maxpooling" should "work correctly" in {
    val initWeight = Tensor[Float](4096, 256 * 6 * 6).rand()
    val initBias = Tensor[Float](4096).rand()
    val input = Tensor[Float](4, 256, 13, 13).rand()

    val dnn = Sequential()
      .add(MaxPooling(3, 3, 2, 2))
      .add(Linear(256 * 6 * 6, 4096, initWeight = initWeight, initBias = initBias))
      .add(ReorderMemory(HeapData(Array(4, 4096), Memory.Format.nc)))
    dnn.compile(TrainingPhase, Array(HeapData(input.size(), Memory.Format.nchw)))

    val blas = nn.Sequential()
      .add(nn.SpatialMaxPooling(3, 3, 2, 2))
      .add(nn.View(256 * 6 * 6))
      .add(nn.Linear(256 * 6 * 6, 4096, initWeight = initWeight, initBias = initBias))

    blas.forward(input)
    dnn.forward(input)

    val gradOutput = Tensor[Float]().resizeAs(blas.output.toTensor).rand()
    dnn.backward(input, gradOutput)
    blas.backward(input, gradOutput)

    Tools.dense(dnn.output) should be (blas.output)
    Tools.dense(dnn.gradInput) should be (blas.gradInput)
  }

//  "relu + linear with 1-D" should "work correctly" in {
//    val initWeight = Tensor(10, 20).rand(-1, 1)
//    val initBias = Tensor(10).rand(-1, 1)
//
//    val input = Tensor(20).rand()
//    val inputFormat = HeapData(Array(20), Memory.Format.x)
//    val outputFormat = HeapData(Array(10), Memory.Format.x)
//
//    val dnn = Sequential().add(ReLU()).add(Linear(20, 10, initWeight = initWeight,
//      initBias = initBias))
//    dnn.compile(TrainingPhase, Array(inputFormat))
//
//    val blas = nn.Sequential().add(nn.ReLU()).add(nn.Linear(20, 10, initWeight = initWeight,
//      initBias = initBias))
//
//    dnn.forward(input)
//    println("=" * 80)
//    blas.forward(input)
//
//    val gradOutput = Tensor().resizeAs(blas.output.toTensor)
//    dnn.backward(input, gradOutput)
//    blas.backward(input, gradOutput)
//  }

//  "1-D input" should "work correctly" in {
//    val input = Tensor(20).rand()
//    val gradOutput = Tensor(10).rand()
//
//    val model = Linear(20, 10)
//    model.setRuntime(new MklDnnRuntime)
//    model.initFwdPrimitives(Array(HeapData(Array(20), Memory.Format.x)), TrainingPhase)
//    model.initBwdPrimitives(Array(HeapData(Array(10), Memory.Format.x)), TrainingPhase)
//    model.initGradWPrimitives(Array(HeapData(Array(10), Memory.Format.x)), TrainingPhase)
//
//    model.forward(input)
//
//    model.updateGradInput(input, gradOutput)
//  }

  "linear + linear, the first linear with a 4-D input" should "work correctly" in {
    val inputSize = 16 * 16 * 16
    val outputSize = 16 * 16 * 16
    val initWeight = Tensor[Float](outputSize, inputSize).rand()
    val initBias = Tensor[Float](outputSize)

    val input = Tensor[Float](16, inputSize).rand()
    val input2 = Tensor[Float](16, 16, 16, 16).rand()

    val inputShape1 = Array(16, inputSize)
    val inputShape2 = Array(16, 16, 16, 16)

    val seq = Sequential()
      .add(Linear(outputSize, inputSize, initWeight = initWeight, initBias = initBias))
      .add(Linear(outputSize, inputSize, initWeight = initWeight, initBias = initBias))

    seq.compile(TrainingPhase, Array(HeapData(inputShape1, Memory.Format.nc)))

    val seq2 = Sequential()
      .add(Linear(outputSize, inputSize, initWeight = initWeight, initBias = initBias))
      .add(Linear(outputSize, inputSize, initWeight = initWeight, initBias = initBias))

    seq2.compile(TrainingPhase, Array(HeapData(inputShape2, Memory.Format.nchw)))

    seq.forward(input)
    seq.backward(input, input)

    seq.forward(input2)
    seq.backward(input2, input)
  }


  "linear " should "work correctly" in {
    val (batchSize, nInput) = (4, 64)
    val inputShape = Array(batchSize, nInput)
    val nOutput = 1000
    val outputShape = Array(batchSize, nOutput)
    val name = "fc"

    val prototxt =
      s"""
         |name: "relu-simple"
         |force_backward: true
         |layer {
         |  name: "data"
         |  type: "DummyData"
         |  top: "data"
         |  include {
         |    phase: TRAIN
         |  }
         |  dummy_data_param {
         |    data_filler {
         |      type: "xavier"
         |    }
         |    shape: { ${shape2Dim(inputShape)} }
         |  }
         |}
         |
         |layer {
         |  bottom: "data"
         |  top: "$name"
         |  name: "$name"
         |  type: "InnerProduct"
         |  inner_product_param {
         |    num_output: $nOutput
         |    weight_filler {
         |      type: "gaussian"
         |      std: 0.01
         |    }
         |    bias_filler {
         |      type: "constant"
         |      value: 0
         |    }
         |  }
         |}
       """.stripMargin
    val linear = new Linear(nInput, nOutput).setName(name)
    linear.setRuntime(new MklDnnRuntime)
    linear.initFwdPrimitives(Array(HeapData(inputShape, Memory.Format.nc)), TrainingPhase)
    linear.initBwdPrimitives(Array(HeapData(outputShape, Memory.Format.nc)), TrainingPhase)
    linear.initGradWPrimitives(Array(HeapData(outputShape, Memory.Format.nc)), TrainingPhase)

    Tools.compare(prototxt, linear, inputShape, outputShape)
  }

  private def shape2Dim(shape: Array[Int]): String = {
    shape.map(x => "dim: " + x).mkString(" ")
  }
}
