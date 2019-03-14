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

import breeze.linalg.reshape
import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.{Module, nn}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, DataFormat, TensorModule}
import com.intel.analytics.bigdl.nn.mkldnn.Phase.{InferencePhase, TrainingPhase}
import com.intel.analytics.bigdl.nn.{Graph, Squeeze, mkldnn}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{BigDLSpecHelper, Engine, RandomGenerator, T}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.numeric.NumericFloat

class BlasWrapperSpec extends BigDLSpecHelper {

  override def doBefore(): Unit = {
     Engine.init(1, 4, true)
  }

  def modelBlas(format: DataFormat = DataFormat("NCHW")) : Module[Float] = {
    val conv1 = nn.SpatialConvolution(1, 20, 5, 5, format = format).inputs()
    val pool1 = nn.SpatialMaxPooling(2, 2, 2, 2, format = format).setName("pool").inputs(conv1)
    val conv2 = nn.SpatialConvolution(20, 50, 5, 5, format = format).inputs(pool1)
    val pool2 = nn.SpatialMaxPooling(2, 2, 2, 2, format = format).inputs(conv2)
    val reshape = nn.Reshape(Array(50 * 4 * 4)).inputs(pool2)
    val fc = nn.Linear(50 * 4 * 4, 500).inputs(reshape)
    val relu = nn.ReLU().setName("relu1").inputs(fc)
    val fc2 = nn.Linear(500, 10).setName("ip2").inputs(relu)
    val log = nn.LogSoftMax().inputs(fc2)
    Graph(conv1, log)
  }

  def modelWrapper(format: Int = Memory.Format.nchw, shape: Array[Int]) : DnnGraph = {
    val input = mkldnn.Input(shape, format).inputs()
    val conv1 = BlasWrapper(nn.SpatialConvolution[Float](1, 20, 5, 5)).inputs(input)
    val pool1 = mkldnn.MaxPooling(2, 2, 2, 2).setName("pool").inputs(conv1)
    val conv2 = BlasWrapper(nn.SpatialConvolution[Float](20, 50, 5, 5)).inputs(pool1)
    val pool2 = mkldnn.MaxPooling(2, 2, 2, 2).inputs(conv2)
    val fc = mkldnn.Linear(50 * 4 * 4, 500).inputs(pool2)
    val relu = mkldnn.ReLU().setName("relu1").inputs(fc)
    val fc2 = mkldnn.Linear(500, 10).setName("ip2").inputs(relu)
    val log = BlasWrapper(nn.LogSoftMax[Float]()).inputs(fc2)
    DnnGraph(Array(input), Array(log))
  }

  "wrapper model" should "be correct" in {
    val inputShape = Array(2, 1, 28, 28)
    val outputShape = Array(2, 10)
    val input = Tensor[Float](inputShape).rand()
    val gradOutput = Tensor[Float](outputShape).rand()

    RandomGenerator.RNG.setSeed(1)
    val blas = modelBlas()
    RandomGenerator.RNG.setSeed(1)
    val wrapper = modelWrapper(Memory.Format.nchw, inputShape)
    wrapper.compile(TrainingPhase)

    val out1 = blas.forward(input)
    val out2 = wrapper.forward(input)

    out1 should be(out2)

    val grad1 = blas.backward(input, gradOutput)
    val grad2 = wrapper.backward(input, gradOutput)

    val weight1 = blas.getParameters()._1
    val weight2 = wrapper.getParameters()._1

    val gradWeight1 = blas.getParameters()._2
    val gradWeight2 = wrapper.getParameters()._2

    grad1 should be(grad2)

    weight1 should be(weight2)
    gradWeight1 should be(gradWeight2)
  }

  "wrapper model run with blas multithread" should "be correct" in {
    val inputShape = Array(4, 1, 28, 28)
    val input = Tensor[Float](inputShape).rand()

    RandomGenerator.RNG.setSeed(1)
    val wrapper = modelWrapper(Memory.Format.nchw, inputShape)
    wrapper.evaluate()
    wrapper.compile(InferencePhase)
    val out1 = wrapper.forward(input)

    RandomGenerator.RNG.setSeed(1)
    System.setProperty("multiThread", "true")
    val wrapperMulti = modelWrapper(Memory.Format.nchw, inputShape)
    wrapperMulti.evaluate()
    wrapperMulti.compile(InferencePhase)
    val out2 = wrapperMulti.forward(input)

    out1 should be(out2)
    System.clearProperty("multiThread")
  }
}
