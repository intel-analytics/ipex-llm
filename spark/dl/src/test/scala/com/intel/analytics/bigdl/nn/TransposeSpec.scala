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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

class TransposeSpec extends FlatSpec with Matchers {
  "A Transpose Module " should "generate correct output" in {
    val input = Tensor[Float]().resize(Array(2, 3)).randn()
    val layer = Transpose[Float](Array((1, 2)))
    var output = layer.updateOutput(input)
    for (i <- 1 to 2; j <- 1 to 3) {
      val a = input.valueAt(i, j)
      val b = output.valueAt(j, i)
      a should be (b)
    }

    output.isContiguous() should be (true)
    input.resize(Array(2, 2)).randn()
    output = layer.updateOutput(input)
    for (i <- 1 to 2; j <- 1 to 2) {
      val a = input.valueAt(i, j)
      val b = output.valueAt(j, i)
      a should be (b)
    }

    input.resize(Array(2, 1, 3)).randn()
    output = layer.updateOutput(input)
    for (i <- 1 to 2; j <- 1 to 1; k <- 1 to 3) {
      val a = input.valueAt(i, j, k)
      val b = output.valueAt(j, i, k)
      a should be (b)
    }
  }

  "A Transpose Module " should "generate correct grad" in {
    val input = Tensor[Float]().resize(Array(2, 3)).randn()
    val gradOutput = Tensor[Float]().resize(Array(2, 3)).randn()
    val layer = Transpose[Float](Array((1, 2)))
    var gradInput = layer.updateGradInput(input, gradOutput)
    for (i <- 1 to 2; j <- 1 to 3) {
      val a = gradOutput.valueAt(i, j)
      val b = gradInput.valueAt(j, i)
      a should be (b)
    }

    gradInput.isContiguous() should be (true)
    gradOutput.resize(Array(2, 2)).randn()
    gradInput = layer.updateGradInput(input, gradOutput)
    for (i <- 1 to 2; j <- 1 to 2) {
      val a = gradOutput.valueAt(i, j)
      val b = gradInput.valueAt(j, i)
      a should be (b)
    }

    gradOutput.resize(Array(2, 3, 5)).randn()
    gradInput = layer.updateGradInput(input, gradOutput)
    for (i <- 1 to 2; j <- 1 to 3; k <- 1 to 5) {
      val a = gradOutput.valueAt(i, j, k)
      val b = gradInput.valueAt(j, i, k)
      a should be (b)
    }

  }
}

class TransposeSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val transpose = Transpose[Float](Array((1, 2))).setName("transpose")
    val input = Tensor[Float]().resize(Array(2, 3)).apply1(_ => Random.nextFloat())
    runSerializationTest(transpose, input)
  }
}
