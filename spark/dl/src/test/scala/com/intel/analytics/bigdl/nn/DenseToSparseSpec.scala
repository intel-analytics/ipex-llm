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
import org.scalatest.{Matchers, FlatSpec}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

@com.intel.analytics.bigdl.tags.Parallel
class DenseToSparseSpec extends FlatSpec with Matchers  {
  "A DenseToSparse forward" should "generate correct output" in {
    val dts = DenseToSparse()
    val denseTensor = Tensor.range(1, 12, 1)
    val output = dts.forward(denseTensor)
    val exceptedOutput = Tensor.sparse(Array(Array.range(0, 12)),
      Array.range(1, 13).map(_.toFloat), Array(12))
    output should be (exceptedOutput)
  }

  "A DenseToSparse backward" should "generate correct output" in {
    val dts = DenseToSparse()
    val input = Tensor.range(1, 12, 1)
    val sparseTensor = Tensor.sparse(Array(Array.range(0, 12)),
      Array.range(1, 13).map(_.toFloat), Array(12))
    val output = dts.backward(input, sparseTensor)
    val exceptedOutput = Tensor.range(1, 12, 1)
    output should be (exceptedOutput)
  }

  // It is useful when DenseToSparse layer is placed on the top of model,
  // received gradient from sparselinear layer.
  "A DenseToSparse backward" should "be able to work without gradOutput" in {
    val mockInput = Tensor[Float](5, 10).rand()
    val mockError = Tensor[Float](5, 10).rand()
    var model = Sequential[Float]()
      .add(DenseToSparse[Float](propagateBack = true))
      .add(SparseLinear[Float](10, 10))
    model.forward(mockInput)
    intercept[Exception] {
      model.backward(mockInput, mockError)
    }

    model = Sequential[Float]()
      .add(DenseToSparse[Float](propagateBack = false))
      .add(SparseLinear[Float](10, 10))
    model.forward(mockInput)
    model.backward(mockInput, mockError)
  }
}

class DenseToSparseSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val denseToSparse = DenseToSparse[Float]().setName("denseToSparse")
    val input = Tensor.range[Float](1, 12, 1)
    runSerializationTest(denseToSparse, input)
  }
}
