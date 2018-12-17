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
package com.intel.analytics.bigdl.nn.ops

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}

class LogicalAndSpec extends FlatSpec with Matchers {
  "LogicalAnd operation" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericBoolean
    val input =
      T(
        Tensor(T(true, false, true)),
        Tensor(T(false, false, true))
      )

    val expectOutput = Tensor(T(false, false, true))

    val output = LogicalAnd().forward(input)
    output should be(expectOutput)
  }
}

class LogicalAndSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val logicalAnd = LogicalAnd[Float].setName("logicalAnd")
    val input = T(Tensor[Boolean](T(true, false)), Tensor[Boolean](T(true, false)))
    runSerializationTest(logicalAnd, input, logicalAnd.
      asInstanceOf[ModuleToOperation[Float]].module.getClass)
  }
}
