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

import com.intel.analytics.bigdl.mkl.MklDnn
import com.intel.analytics.bigdl.nn.{Concat => Concat2, Identity}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.scalatest.{FlatSpec, Matchers}

import scala.collection.mutable.ArrayBuffer


class ConcatDnnSpec extends FlatSpec with Matchers {
  "ConcatDnn" should "work correctly" in {
    val model = ConcatDnn(2)
    model.add(Identity[Float]())
    model.add(Identity[Float]())

    val model2 = Concat2[Float](2)
    model2.add(Identity[Float]())
    model2.add(Identity[Float]())

    val input = Tensor[Float](2, 8, 3, 4).randn()
    val gradOutput = Tensor[Float](2, 16, 3, 4).randn()
    val output = model.forward(input)
    val output2 = model2.forward(input)
    val gradInput = model.backward(input, gradOutput)
    val gradInput2 = model2.backward(input, gradOutput)

    DnnUtils.nearequals(output, output2)
    DnnUtils.nearequals(gradInput, gradInput2)
  }
}
