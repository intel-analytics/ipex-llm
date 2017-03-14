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

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.scalatest.{FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Parallel
class CopySpec extends FlatSpec with Matchers {
  "A Copy" should "generate correct output" in {
    val output = Tensor[Double](Storage[Double](Array(
      2.7183, 7.3891, 20.0855,
      54.5982, 148.4132, 403.4288)), 1, Array(2, 3))

    val input = Tensor[Double](Storage[Double](Array(
      2.7183, 7.3891f, 20.0855f,
      54.5982f, 148.4132f, 403.4288f)), 1, Array(2, 3))

    val copy = new Copy[Double]()

    val copyOutput = copy.forward(input)

    copyOutput should equal (output)
  }

  "A Copy" should "generate correct grad" in {
    val input = Tensor(Storage[Double](Array(1.0, 2, 3, 4, 5, 6)), 1, Array(2, 3))

    val gradOutput = Tensor(Storage(Array(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)), 1, Array(2, 3))

    val copy = new Copy[Double]()

    val output = copy.forward(input)
    val gradInput = copy.backward(input, gradOutput)

    output should equal (input)
    gradInput should equal (gradOutput)
  }
}
