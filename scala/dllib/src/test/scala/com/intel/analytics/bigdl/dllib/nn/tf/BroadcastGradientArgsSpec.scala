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

import com.intel.analytics.bigdl.nn.ops.ModuleToOperation
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}

class BroadcastGradientArgsSpec extends FlatSpec with Matchers {
  "BroadcastGradientArgs operation" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericInt
    val input =
      T(
        Tensor(T(1, 2, 3)),
        Tensor(T(2, 2, 1))
      )

    val expectOutput = T(Tensor(T(0)), Tensor(T(2)))

    val output = new BroadcastGradientArgs().forward(input)
    output should be(expectOutput)
  }

  "BroadcastGradientArgs operation empty" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericInt
    val input =
      T(
        Tensor(T(1, 2, 3)),
        Tensor()
      )

    val expectOutput = T(Tensor(T(0)), Tensor(T(0, 1, 2)))

    val output = new BroadcastGradientArgs().forward(input)
    output should be(expectOutput)
  }
}

class BroadcastGradientArgsSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val broadcastGradientArgs = BroadcastGradientArgs[Float]().
      setName("broadcastGradientArgs")
    val input =
      T(
        Tensor[Int](T(1, 2, 3)),
        Tensor[Int](T(2, 2, 1))
      )
    runSerializationTest(broadcastGradientArgs, input, broadcastGradientArgs.
      asInstanceOf[ModuleToOperation[Float]].module.getClass)
  }
}
