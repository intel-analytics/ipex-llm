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
package com.intel.analytics.sparkdl.nn

import com.intel.analytics.sparkdl.tensor.{Storage, Tensor}
import org.scalatest.{FlatSpec, Matchers}

class SquareSpec extends FlatSpec with Matchers {
  "A Square" should "generate correct output when input is contiguous" in {
    val input = Tensor(Storage[Double](Array(1.0, 2, 3, 4, 5, 6)), 1, Array(2, 3))

    val output = Tensor(Storage(Array(
      1.0, 4, 9,
      16, 25, 36)), 1, Array(2, 3))

    val square = new Square[Double]()

    val squareOutput = square.forward(input)

    squareOutput should equal (output)
  }

  "A Square" should "generate correct output is not contiguous" in {
    val input = Tensor(Storage[Double](Array(1.0, 2, 3, 4, 5, 6)), 1, Array(2, 3)).t()

    val output = Tensor(Storage(Array(
      1.0, 4, 9,
      16, 25, 36)), 1, Array(2, 3)).t()

    val square = new Square[Double]()

    val squareOutput = square.forward(input)

    squareOutput should equal (output)
  }

  "A Square" should "generate correct gradInput is contiguous" in {
    val input = Tensor(Storage[Double](Array(1.0, 2, 3, 4, 5, 6)), 1, Array(2, 3))

    val gradOutput = Tensor(Storage(Array(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)), 1, Array(2, 3))

    val square = new Square[Double]()

    val output = square.forward(input)
    val gradInput = square.backward(input, gradOutput)

    gradInput should equal (Tensor(Storage(Array(0.2, 0.8, 1.8, 3.2, 5.0, 7.2)), 1, Array(2, 3)))
  }

  "A Square" should "generate correct gradInput is not contiguous" in {
    val input = Tensor(Storage[Double](Array(1.0, 2, 3, 4, 5, 6)), 1, Array(2, 3)).t()

    val gradOutput = Tensor(Storage(Array(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)), 1, Array(2, 3)).t()

    val square = new Square[Double]()

    val output = square.forward(input)
    val gradInput = square.backward(input, gradOutput)

    gradInput should equal (Tensor(
      Storage(Array(0.2, 0.8, 1.8, 3.2, 5.0, 7.2)), 1, Array(2, 3)).t())
  }
}
