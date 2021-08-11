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
package com.intel.analytics.bigdl.nn.tf

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

@com.intel.analytics.bigdl.tags.Parallel
class StridedSliceSpec extends FlatSpec with Matchers {

  "StrideSlice " should "compute correct output and gradient" in {
    val module1 = new StridedSlice[Double, Double]()
    val inputTensor = Tensor[Double](2, 2, 2)
    inputTensor(Array(1, 1, 1)) = -0.17020166106522
    inputTensor(Array(1, 1, 2)) = 0.57785657607019
    inputTensor(Array(1, 2, 1)) = -1.3404131438583
    inputTensor(Array(1, 2, 2)) = 1.0938102817163
    inputTensor(Array(2, 1, 1)) = 1.120370157063
    inputTensor(Array(2, 1, 2)) = -1.5014141565189
    inputTensor(Array(2, 2, 1)) = 0.3380249235779
    inputTensor(Array(2, 2, 2)) = -0.625677742064
    val begin = Tensor[Int](3).fill(1)
    val end = Tensor[Int](3).fill(3)
    end.setValue(1, 2)
    val strides = Tensor[Int](3).fill(1)

    val expectOutput1 = Tensor[Double](1, 2, 2)
    expectOutput1(Array(1, 1, 1)) = -0.17020166106522
    expectOutput1(Array(1, 1, 2)) = 0.57785657607019
    expectOutput1(Array(1, 2, 1)) = -1.3404131438583
    expectOutput1(Array(1, 2, 2)) = 1.0938102817163

    val expectedGradInput = Tensor[Double](2, 2, 2)
    expectedGradInput(Array(1, 1, 1)) = -0.17020166106522
    expectedGradInput(Array(1, 1, 2)) = 0.57785657607019
    expectedGradInput(Array(1, 2, 1)) = -1.3404131438583
    expectedGradInput(Array(1, 2, 2)) = 1.0938102817163
    expectedGradInput(Array(2, 1, 1)) = 0.0
    expectedGradInput(Array(2, 1, 2)) = 0.0
    expectedGradInput(Array(2, 2, 1)) = 0.0
    expectedGradInput(Array(2, 2, 2)) = 0.0

    val input = T(inputTensor, begin, end, strides)
    val output1 = module1.forward(input)
    val gradInput = module1.backward(input, output1).toTable[Tensor[Double]](1)

    output1 should be(expectOutput1)
    gradInput should be(expectedGradInput)
  }

}

class StridedSliceSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val stridedSlice = StridedSlice[Float, Float]().setName("stridedSlice")
    val input = Tensor[Float](2, 2, 2).apply1(_ => Random.nextFloat())
    val begin = Tensor[Int](3).fill(1)
    val end = Tensor[Int](3).fill(3)
    end.setValue(1, 2)
    val strides = Tensor[Int](3).fill(1)
    runSerializationTest(stridedSlice, T(input, begin, end, strides))
  }
}
