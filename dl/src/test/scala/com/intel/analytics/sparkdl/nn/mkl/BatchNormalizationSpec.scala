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

class BatchNormalizationSpec extends FlatSpec with Matchers {
  "BatchNormalization output and gradInput compared with caffe" should "are the same" in {
    val modelDnn  = new SpatialBatchNormalization[Float](64, 1e-3)
    val modelBlas = new nn.SpatialBatchNormalization[Float](64, 1e-3)

    val input = Tools.GetTensorFloat("input", Array(32, 64, 112, 112))
    val weights = Tools.GetTensorFloat("weights", Array(64))
    val bias = Tools.GetTensorFloat("bias", Array(64))

    modelDnn.weight.set(weights)
    modelDnn.bias.set(bias)
    modelBlas.weight.set(weights)
    modelBlas.bias.set(bias)

    modelDnn.forward(input)
    modelBlas.forward(input)

    val output = Tools.GetTensorFloat("output", modelDnn.output.size())

    Tools.PrintTensor(modelDnn.output, msg = "dnn output")
    Tools.PrintTensor(output, msg = "caffe output")
    Tools.AverageAll(modelDnn.output, "dnn output")
    Tools.AverageAll(output, "caffe output")

    val gradOutput = Tools.GetTensorFloat("gradOutput", output.size())
    val gradInput  = Tools.GetTensorFloat("gradInput", input.size())

    modelDnn.backward(input, gradOutput)
    modelBlas.backward(input, gradOutput)

    Tools.PrintTensor(modelDnn.gradInput, msg = "dnn gradinput")
    Tools.PrintTensor(gradInput, msg = "blas gradinput")
    Tools.AverageAll(modelDnn.gradInput, "dnn gradient input")
    Tools.AverageAll(gradInput, "blas gradient input")

    Tools.CumulativeError(modelDnn.output, output, "output") should be(0.0 +- 1e-6)
    Tools.CumulativeError(modelDnn.gradInput, gradInput, "gradient input") should be(0.0 +- 1e-6)

    Tools.CumulativeError(modelDnn.output, modelBlas.output, "output")
    Tools.CumulativeError(modelDnn.gradInput, modelBlas.gradInput, "gradient input")
  }
}
