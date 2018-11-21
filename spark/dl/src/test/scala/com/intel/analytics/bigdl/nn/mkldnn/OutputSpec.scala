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

import breeze.numerics.log
import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.nn.{Graph, mkldnn}
import com.intel.analytics.bigdl.{nn, _}
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.nn.mkldnn.Phase.TrainingPhase
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.{BigDLSpecHelper, RandomGenerator}

class OutputSpec extends BigDLSpecHelper {
  def model() : DnnGraph = {
    val conv1 = mkldnn.SpatialConvolution(1, 20, 5, 5).inputs()
    val pool1 = mkldnn.MaxPooling(2, 2, 2, 2).setName("pool").inputs(conv1)
    val conv2 = mkldnn.SpatialConvolution(20, 50, 5, 5).inputs(pool1)
    val out = mkldnn.Output(Memory.Format.nchw).inputs(conv2)
    DnnGraph(Array(conv1), Array(out))
  }

  def blas() : Module[Float] = {
    val conv1 = nn.SpatialConvolution(1, 20, 5, 5).inputs()
    val pool1 = nn.SpatialMaxPooling(2, 2, 2, 2).setName("pool").inputs(conv1)
    val conv2 = nn.SpatialConvolution(20, 50, 5, 5).inputs(pool1)
    Graph(conv1, conv2)
  }

  "test output" should "be right" in {
    val inputShape = Array(2, 1, 28, 28)
    val outputShape = Array(2, 50, 8, 8)
    val input = Tensor[Float](inputShape).rand()
    val gradOutput = Tensor[Float](outputShape).rand()

    RandomGenerator.RNG.setSeed(1)
    val modelDnn = model()
    RandomGenerator.RNG.setSeed(1)
    val modelBlas = blas()

    modelDnn.setRuntime(new MklDnnRuntime)
    modelDnn.initFwdPrimitives(Array(HeapData(inputShape, Memory.Format.nchw)), TrainingPhase)
    modelDnn.initBwdPrimitives(Array(HeapData(outputShape, Memory.Format.nchw)), TrainingPhase)
    modelDnn.initGradWPrimitives(Array(HeapData(outputShape, Memory.Format.nchw)), TrainingPhase)

    val out1 = modelBlas.forward(input)
    val out2 = modelDnn.forward(input)

    out1 should be(out2)

    val grad1 = modelBlas.backward(input, gradOutput)
    val grad2 = modelDnn.backward(input, gradOutput).toTensor[Float]
    val gradInputTensor = Tensor[Float]().resize(grad2.size()).copy(grad2)

    grad1 should be(gradInputTensor)

    val weight1 = modelBlas.getParameters()._1
    val weight2 = modelDnn.getParameters()._1

    val gradWeight1 = modelBlas.getParameters()._2
    val gradWeight2 = modelDnn.getParameters()._2

    weight1 should be(weight2)
    gradWeight1 should be(gradWeight2)
  }
}
