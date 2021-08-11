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
package com.intel.analytics.bigdl.tensor

import org.scalatest.{FlatSpec, Matchers}

class TensorSpec extends FlatSpec with Matchers {

  "Tensor factory method" should "be able to construct scalar" in {
    val tensor = Tensor[Int](Array(4), Array[Int]())
    tensor.value() should be (4)
    tensor.size() should be (Array[Int]())
    tensor.nDimension() should be (0)
    tensor.isScalar should be (true)
  }

  "Tensor resize " should "work for scalar" in {
    val tensor = Tensor[Int]()
    tensor.resize(Array[Int]())
    tensor.value() should be (0)
    tensor.size() should be (Array[Int]())
    tensor.nDimension() should be (0)
    tensor.isScalar should be (true)
  }

  "Tensor resizeAs " should "work for scalar" in {
    val tensor = Tensor[Int]()
    val tensorScalar = Tensor[Int](Array(1), Array[Int]())
    tensor.resizeAs(tensorScalar)
    tensor.value() should be (0)
    tensor.size() should be (Array[Int]())
    tensor.nDimension() should be (0)
    tensor.isScalar should be (true)
  }

  "Tensor set " should "work for scalar" in {
    val tensor = Tensor[Int]()
    tensor.resize(Array[Int](1, 2))
    tensor.set()
    tensor.isEmpty should be (true)
  }

  "zero Tensor set " should "work" in {
    val tensor = Tensor[Int](0)
    tensor.set()
    tensor.isEmpty should be (true)
  }

  "Empty Tensor set " should "work" in {
    val tensor = Tensor[Int]()
    tensor.set()
    tensor.isEmpty should be (true)
  }

}
