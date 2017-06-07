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

import com.intel.analytics.bigdl.optim.{L2Regularizer, SGD}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Parallel
class SpatialFullConvolutionSpec extends FlatSpec with Matchers {
  "SpatialFullConvolution L2 regularizer" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble

    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 2
    val kH = 2
    val dW = 1
    val dH = 1
    val padW = 0
    val padH = 0

    val inputData = Array(
      1.0, 2, 3,
      4, 5, 6,
      7, 8, 9
    )

    val state1 = T("learningRate" -> 0.1, "learningRateDecay" -> 5e-7,
      "weightDecay" -> 0.1, "momentum" -> 0.002)
    val state2 = T("learningRate" -> 0.1, "learningRateDecay" -> 5e-7,
      "weightDecay" -> 0.0, "momentum" -> 0.002)

    val inputN = 5
    val outputN = 2
    val batchSize = 5
    val criterion = new MSECriterion[Double]

    val input = Tensor[Double](Storage(inputData), 1, Array(1, 3, 3))
    val labels = Tensor[Double](16).rand()

    val model1 = Sequential()
      .add(SpatialFullConvolution[Tensor[Double], Double](nInputPlane, nOutputPlane,
        kW, kH, dW, dH, padW, padH))
      .add(Sigmoid())
    val (weights1, grad1) = model1.getParameters()

    val model2 = Sequential()
      .add(SpatialFullConvolution[Tensor[Double], Double](nInputPlane, nOutputPlane,
        kW, kH, dW, dH, padW, padH,
        wRegularizer = L2Regularizer(0.1), bRegularizer = L2Regularizer(0.1)))
      .add(Sigmoid())
    val (weights2, grad2) = model2.getParameters()
    weights2.copy(weights1.clone())
    grad2.copy(grad1.clone())


    val sgd = new SGD[Double]

    def feval1(x: Tensor[Double]): (Double, Tensor[Double]) = {
      val output = model1.forward(input).toTensor[Double]
      val _loss = criterion.forward(output, labels)
      model1.zeroGradParameters()
      val gradInput = criterion.backward(output, labels)
      model1.backward(input, gradInput)
      (_loss, grad1)
    }

    def feval2(x: Tensor[Double]): (Double, Tensor[Double]) = {
      val output = model2.forward(input).toTensor[Double]
      val _loss = criterion.forward(output, labels)
      model2.zeroGradParameters()
      val gradInput = criterion.backward(output, labels)
      model2.backward(input, gradInput)
      (_loss, grad2)
    }

    var loss1: Array[Double] = null
    for (i <- 1 to 100) {
      loss1 = sgd.optimize(feval1, weights1, state1)._2
      println(s"${i}-th loss = ${loss1(0)}")
    }

    var loss2: Array[Double] = null
    for (i <- 1 to 100) {
      loss2 = sgd.optimize(feval2, weights2, state2)._2
      println(s"${i}-th loss = ${loss2(0)}")
    }


    weights1 should be(weights2)
    loss1 should be(loss2)
  }

  "A SpatialFullConvolution BilinearFiller" should "generate correct parameter" in {
    val conv = new SpatialFullConvolution[Tensor[Double], Double](3, 6, 3, 3, 2, 2,
      0, 0, 0, 0, 1, false)
    conv.reset(weightInitMethod = BilinearFiller, biasInitMethod = Zeros)

    val caffeWeight = Tensor(Storage(Array(
      0.0625, 0.1875, 0.1875, 0.1875, 0.5625, 0.5625, 0.1875, 0.5625, 0.5625,
      0.0625, 0.1875, 0.1875, 0.1875, 0.5625, 0.5625, 0.1875, 0.5625, 0.5625,
      0.0625, 0.1875, 0.1875, 0.1875, 0.5625, 0.5625, 0.1875, 0.5625, 0.5625,
      0.0625, 0.1875, 0.1875, 0.1875, 0.5625, 0.5625, 0.1875, 0.5625, 0.5625,
      0.0625, 0.1875, 0.1875, 0.1875, 0.5625, 0.5625, 0.1875, 0.5625, 0.5625,
      0.0625, 0.1875, 0.1875, 0.1875, 0.5625, 0.5625, 0.1875, 0.5625, 0.5625,
      0.0625, 0.1875, 0.1875, 0.1875, 0.5625, 0.5625, 0.1875, 0.5625, 0.5625,
      0.0625, 0.1875, 0.1875, 0.1875, 0.5625, 0.5625, 0.1875, 0.5625, 0.5625,
      0.0625, 0.1875, 0.1875, 0.1875, 0.5625, 0.5625, 0.1875, 0.5625, 0.5625,
      0.0625, 0.1875, 0.1875, 0.1875, 0.5625, 0.5625, 0.1875, 0.5625, 0.5625,
      0.0625, 0.1875, 0.1875, 0.1875, 0.5625, 0.5625, 0.1875, 0.5625, 0.5625,
      0.0625, 0.1875, 0.1875, 0.1875, 0.5625, 0.5625, 0.1875, 0.5625, 0.5625,
      0.0625, 0.1875, 0.1875, 0.1875, 0.5625, 0.5625, 0.1875, 0.5625, 0.5625,
      0.0625, 0.1875, 0.1875, 0.1875, 0.5625, 0.5625, 0.1875, 0.5625, 0.5625,
      0.0625, 0.1875, 0.1875, 0.1875, 0.5625, 0.5625, 0.1875, 0.5625, 0.5625,
      0.0625, 0.1875, 0.1875, 0.1875, 0.5625, 0.5625, 0.1875, 0.5625, 0.5625,
      0.0625, 0.1875, 0.1875, 0.1875, 0.5625, 0.5625, 0.1875, 0.5625, 0.5625,
      0.0625, 0.1875, 0.1875, 0.1875, 0.5625, 0.5625, 0.1875, 0.5625, 0.5625
    )), 1, Array(1, 3, 6, 3, 3))

    conv.weight should be (caffeWeight)
  }

  "A SpatialFullConvolution BilinearFiller(1, 2, 4, 4)" should "generate correct parameter" in {
    val conv = new SpatialFullConvolution[Tensor[Double], Double](1, 2, 4, 4, 2, 2,
      0, 0, 0, 0, 1, false)
    conv.reset(weightInitMethod = BilinearFiller, biasInitMethod = Zeros)

    val caffeWeight = Tensor(Storage(Array(
      0.0625, 0.1875, 0.1875, 0.0625,
      0.1875, 0.5625, 0.5625, 0.1875,
      0.1875, 0.5625, 0.5625, 0.1875,
      0.0625, 0.1875, 0.1875, 0.0625,

      0.0625, 0.1875, 0.1875, 0.0625,
      0.1875, 0.5625, 0.5625, 0.1875,
      0.1875, 0.5625, 0.5625, 0.1875,
      0.0625, 0.1875, 0.1875, 0.0625
    )), 1, Array(1, 1, 2, 4, 4))

    conv.weight should be (caffeWeight)
  }
}
