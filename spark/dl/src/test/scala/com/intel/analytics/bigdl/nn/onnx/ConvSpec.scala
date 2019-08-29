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

package com.intel.analytics.bigdl.nn.onnx

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}


class ConvSpec extends FlatSpec with Matchers {

  "Conv" should "work" in {
    val nInputDims = 3
    val nOutputDims = 7
    val kernelShape = List(2, 2)
    val weight = Tensor[Float](84).rand()
    val bias = Tensor[Float](7).rand()
    val conv = Conv[Float](nInputPlane = nInputDims, nOutputPlane = nOutputDims,
      kernelShape = kernelShape, weight = weight, bias = bias)
    val input = Tensor[Float](1, 3, 224, 224).rand()
    val out = conv.forward(input)

    // Todo: verify output
    println(out.toString)

  }
}


class ConvSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val nInputDims = 3
    val nOutputDims = 7
    val kernelShape = List(2, 2)
    val weight = Tensor[Float](84).rand()
    val bias = Tensor[Float](7).rand()
    val conv = Conv[Float](nInputPlane = nInputDims, nOutputPlane = nOutputDims,
      kernelShape = kernelShape, weight = weight, bias = bias)
    val input = Tensor[Float](1, 3, 224, 224).rand()
    runSerializationTest(conv, input)
  }
}
