/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.sparkdl.nn.mkl

import com.intel.analytics.sparkdl.nn
import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.tensor.Tensor

import scala.reflect.ClassTag

class LRNSpec extends FlatSpec with Matchers {
  "LRN output and gradient input" should "generate correct result" in {
    def test[T: ClassTag]()(implicit ev: TensorNumeric[T]) {
      val modelDnn  = new LocalNormalizationAcrossChannels[T](5, 0.0001, 0.75)
      val modelBlas = new nn.LocalNormalizationAcrossChannels[T](5, 0.0001, 0.75)

      for (i <- 0 until Tools.GetRandTimes()) {
        val input = Tensor[T](Array(32, 64, 112, 112)).fill(ev.fromType(0.1))

        modelDnn.forward(input)
        modelBlas.forward(input)

        Tools.PrintTensor(modelDnn.output, msg = "dnn output")
        Tools.PrintTensor(modelBlas.output, msg = "blas output")
        Tools.AverageAll(modelDnn.output, "dnn output")
        Tools.AverageAll(modelBlas.output, "blas output")

        val gradOutput = Tensor[T]().resizeAs(modelDnn.output).fill(ev.fromType(0.1))

        modelDnn.backward(input, gradOutput)
        modelBlas.backward(input, gradOutput)

        Tools.PrintTensor(modelDnn.gradInput, msg = "dnn gradinput")
        Tools.PrintTensor(modelBlas.gradInput, msg = "blas gradinput")
        Tools.AverageAll(modelDnn.gradInput, "dnn gradient input")
        Tools.AverageAll(modelBlas.gradInput, "blas gradient input")
        Tools.CumulativeError(modelDnn.output, modelBlas.output, "output") should be(0.0 +- 1e-6)
        Tools.CumulativeError(modelDnn.gradInput, modelBlas.gradInput, "gradient input") should be(0.0 +- 1e-6)
      }
    }

    test[Float]()
  }

  "LRN output and gradient input compared with caffe" should "is right" in {
    val modelDnn = new LocalNormalizationAcrossChannels[Float](5, 0.0001, 0.75)

    val input = Tools.GetTensorFloat("input", Array(32, 64, 112, 112))
    modelDnn.forward(input)
    val output = Tools.GetTensorFloat("output", modelDnn.output.size())

    Tools.PrintTensor(modelDnn.output, msg = "dnn output")
    Tools.PrintTensor(output, msg = "caffe output")
    Tools.AverageAll(modelDnn.output, "dnn output")
    Tools.AverageAll(output, "caffe output")

    val gradOutput = Tools.GetTensorFloat("gradOutput", output.size())
    val gradInput  = Tools.GetTensorFloat("gradInput", input.size())

    modelDnn.backward(input, gradOutput)

    Tools.PrintTensor(modelDnn.gradInput, msg = "dnn gradinput")
    Tools.PrintTensor(gradInput, msg = "blas gradinput")
    Tools.AverageAll(modelDnn.gradInput, "dnn gradient input")
    Tools.AverageAll(gradInput, "blas gradient input")

    Tools.CumulativeError(modelDnn.output, output, "output") should be(0.0 +- 1e-6)
    Tools.CumulativeError(modelDnn.gradInput, gradInput, "gradient input") should be(0.0 +- 1e-6)
  }
}
