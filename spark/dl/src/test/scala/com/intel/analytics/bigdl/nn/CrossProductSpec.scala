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
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}

class CrossProductSpec extends FlatSpec with Matchers {

  // forward result: [9, 10, 25, 18, 45, 50]
  private val input1D = T(
    Tensor[Float](T(1.0f, 2.0f)),
    Tensor[Float](T(3.0f, 3.0f)),
    Tensor[Float](T(2.0f, 4.0f)),
    Tensor[Float](T(5.0f, 10.0f))
  )

  /**
   * backward result with inputSample: T([27, 51], [27, 54], [31, 53], [14, 16])
   * {{{
   *   gradInput(1) = [3, 3] * 1 + [2, 4] * 2 + [5, 10] * 3 = [22, 41]
   *   gradInput(2) = [1, 2] * 1 + [2, 4] * 3 + [5, 10] * 4 = [27, 54]
   *   gradInput(3) = [1, 2] * 2 + [3, 3] * 3 + [5, 10] * 10 = [61, 113]
   *   gradInput(4) = [1, 2] * 3 + [3, 3] * 4 + [2, 4] * 10 = [35, 58]
   * }}}
   */
  private val gradOut1D = Tensor[Float](T(T(1f, 2f, 3f, 3f, 4f, 10f)))


  /**
   * forward result: {[25, 12], [43, 18], [64, 36]}
   * {{{
   *   row1([25, 43, 64]):
   *     2*1 + 2*4 + 3*5 = 25
   *     2*5 + 2*6 + 3*7 = 43
   *     1*5 + 4*6 + 5*7 = 64
   *   row2([12, 18, 30]):
   *     1*2 + 1*2 + 2*4 = 12
   *     1*3 + 1*3 + 2*6 = 18
   *     2*3 + 2*3 + 4*6 = 36
   * }}}
   */
  private val input2D = T(
    Tensor[Float](T(T(2f, 2f, 3f), T(1f, 1f, 2f))),
    Tensor[Float](T(T(1f, 4f, 5f), T(2f, 2f, 4f))),
    Tensor[Float](T(T(5f, 6f, 7f), T(3f, 3f, 6f)))
  )

  /**
   * backward result with inputSample:
   * {{{
   *   gradInput(1,1) = [1, 4, 5] * 2 + [5, 6, 7] * 3 = [17, 26, 31]
   *   gradInput(2,1) = [2, 2, 3] * 2 + [5, 6, 7] * 5 = [29, 34, 41]
   *   gradInput(3,1) = [2, 2, 3] * 3 + [1, 4, 5] * 5 = [11, 26, 34]
   *   gradInput(1,2) = [2, 2, 4] * 4 + [3, 3, 6] * 6 = [26, 26, 52]
   *   gradInput(2,2) = [1, 1, 2] * 4 + [3, 3, 6] * 8 = [28, 28, 56]
   *   gradInput(3,2) = [1, 1, 2] * 6 + [2, 2, 4] * 8 = [22, 22, 44]
   * }}}
   */
  private val gradOut2D = Tensor[Float](T(T(2f, 3f, 5f), T(4f, 6f, 8f)))



  "A CrossProduct" should "thrown Exceptions when inputs are invalid" in {
    intercept[java.lang.IllegalArgumentException] {
      CrossProduct[Float](numTensor = 2).updateOutput(input1D)
    }
    intercept[java.lang.IllegalArgumentException] {
      CrossProduct[Float](embeddingSize = 9).updateOutput(input1D)
    }
  }

  "A CrossProduct.updateOutput" should "work correctly" in {
    val module = CrossProduct[Float]()
    var output = module.forward(
      T(Tensor[Float](T(1.5f, 2.5f)), Tensor[Float](T(1.5f, 3.0f))))
    output.storage().toArray shouldEqual Array(9.75f)

    output = module.forward(input1D)
    output.size() shouldEqual Array(1, 6)
    output.storage().array() shouldEqual Array(9f, 10f, 25f, 18f, 45f, 50f)

    output = module.forward(input2D)
    output.select(1, 1).toArray() shouldEqual Array(25, 43, 64)
    output.select(1, 2).toArray() shouldEqual Array(12, 18, 36)
  }

  "A CrossProduct.updateGradInput" should "work correctly" in {
    var module = CrossProduct[Float]()
    var gradIn = module.backward(input1D, gradOut1D)
    gradIn[Tensor[Float]](1).toArray() shouldEqual Array(22f, 41f)
    gradIn[Tensor[Float]](2).toArray() shouldEqual Array(27f, 54f)
    gradIn[Tensor[Float]](3).toArray() shouldEqual Array(61f, 113f)
    gradIn[Tensor[Float]](4).toArray() shouldEqual Array(35f, 58f)

    gradIn = module.backward(input2D, gradOut2D)
    var t = gradIn[Tensor[Float]](1)
    t.select(1, 1).toArray() shouldEqual Array(17, 26, 31)
    t.select(1, 2).toArray() shouldEqual Array(26, 26, 52)
    t = gradIn[Tensor[Float]](2)
    t.select(1, 1).toArray() shouldEqual Array(29, 34, 41)
    t.select(1, 2).toArray() shouldEqual Array(28, 28, 56)
    t = gradIn[Tensor[Float]](3)
    t.select(1, 1).toArray() shouldEqual Array(11, 26, 34)
    t.select(1, 2).toArray() shouldEqual Array(22, 22, 44)
  }

}

class CrossProductSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val crossProd = CrossProduct[Float]()
    val input = T(Tensor[Float](T(1.0f, 2.0f)),
      Tensor[Float](T(2.0f, 3.0f)), Tensor[Float](T(3.0f, 4.0f)))
    runSerializationTest(crossProd, input)
  }
}
