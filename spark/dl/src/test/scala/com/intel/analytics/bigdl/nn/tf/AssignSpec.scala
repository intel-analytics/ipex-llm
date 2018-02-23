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

class AssignSpec extends FlatSpec with Matchers {
  "Assign operation Float" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericFloat
    val input =
      T(
        Tensor(T(1f, 2f, 3f)),
        Tensor(T(2f, 2f, 4f))
      )

    val expectOutput = Tensor(T(2f, 2f, 4f))

    val output = new Assign().forward(input)
    output should be(expectOutput)
  }

  "Assign operation Double" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericFloat
    val input =
      T(
        Tensor(T(1.0f, 2.0f, 3.0f)),
        Tensor(T(2.0f, 2.0f, 4.0f))
      )

    val expectOutput = Tensor(T(2.0f, 2.0f, 4.0f))

    val output = new Assign[Double]().forward(input)
    output should be(expectOutput)
  }
}

class AssignSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val assign = new Assign[Float]().setName("assign")
    val input =
      T(
        Tensor[Float](T(1f, 2f, 3f)),
        Tensor[Float](T(2f, 2f, 4f))
      )
    runSerializationTest(assign, input)
  }
}
