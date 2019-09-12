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

package com.intel.analytics.bigdl.nn.onnx

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}


class UnsqueezeSpec extends FlatSpec with Matchers {
  "Unsqueeze" should "work" in {
    val unsqueeze = Unsqueeze[Float](List(2), 3)
    val input = T(
      Tensor[Float](2, 3, 5).rand(),
      Tensor[Float](10, 3, 5).rand()
    )

    val out = unsqueeze.forward(Tensor[Float](3, 4, 5).rand())

    println(out.size().mkString(" "))
  }
}


class UnsqueezeSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val unsqueeze = Unsqueeze[Float](List(0), 3)
    val input = T(
      Tensor[Float](2, 3, 5).rand(),
      Tensor[Float](10, 3, 5).rand()
    )
    runSerializationTest(unsqueeze, input)
  }
}
