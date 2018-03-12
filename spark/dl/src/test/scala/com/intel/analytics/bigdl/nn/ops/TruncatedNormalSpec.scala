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


class TruncatedNormalSpec  extends FlatSpec with Matchers {
  "TruncatedNormal operation" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericFloat
    val input =
      Tensor[Int](T(1, 2, 3))

    val output = TruncatedNormal(10, 20).forward(input)
  }
}

class TruncatedNormalSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val truncateNormal = TruncatedNormal[Float, Float](10, 20).setName("truncateNormal")
    val input = Tensor[Int](T(1, 2, 3))
    runSerializationTest(truncateNormal, input, truncateNormal.
      asInstanceOf[ModuleToOperation[Float]].module.getClass)
  }
}
