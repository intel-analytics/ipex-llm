/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.pipeline.api.keras.layers

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.nn.{Squeeze => BSqueeze}
import com.intel.analytics.zoo.pipeline.api.keras.layers.{Squeeze => ZSqueeze}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest


class SqueezeSpec extends ZooSpecHelper {

  "Squeeze one dimension Zoo" should "be the same as BigDL" in {
    val blayer = BSqueeze[Float](dim = 2)
    val zlayer = ZSqueeze[Float](1, inputShape = Shape(1, 4, 3))
    zlayer.build(Shape(-1, 1, 4, 3))
    zlayer.getOutputShape().toSingle().toArray should be (Array(-1, 4, 3))
    val input = Tensor[Float](Array(2, 1, 4, 3)).rand()
    compareOutputAndGradInput(
      blayer.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]], zlayer, input)
  }

  "Squeeze two dimensions Zoo" should "be the same as BigDL" in {
    val blayer = BSqueeze[Float](dims = Array(1, 2), batchMode = true)
    val zlayer = ZSqueeze[Float](dims = Array(1, 2), inputShape = Shape(1, 1, 3))
    zlayer.build(Shape(-1, 1, 1, 3))
    zlayer.getOutputShape().toSingle().toArray should be (Array(-1, 3))
    val input = Tensor[Float](Array(1, 1, 1, 3)).rand()
    compareOutputAndGradInput(
      blayer.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]], zlayer, input)
  }

  "Squeeze all singleton dimensions Zoo" should "be the same as BigDL" in {
    val blayer = BSqueeze[Float](dims = null, batchMode = true)
    val zlayer = ZSqueeze[Float](dims = null, inputShape = Shape(1, 1, 4, 1))
    zlayer.build(Shape(-1, 1, 1, 4, 1))
    zlayer.getOutputShape().toSingle().toArray should be (Array(-1, 4))
    val input = Tensor[Float](Array(2, 1, 1, 4, 1)).rand()
    compareOutputAndGradInput(
      blayer.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]], zlayer, input)
  }

  "Squeeze the batch dimension" should "raise an exception" in {
    intercept[RuntimeException] {
      val zlayer = ZSqueeze[Float](0, inputShape = Shape(1, 1, 3))
      zlayer.build(Shape(-1, 1, 1, 3))
    }
  }

  "Squeeze a non-singleton dimension" should "raise an exception" in {
    intercept[RuntimeException] {
      val klayer = ZSqueeze[Float](1, inputShape = Shape(2, 1, 3))
      klayer.build(Shape(-1, 2, 1, 3))
    }
  }

  "Squeeze a negative dimension index" should "raise an exception" in {
    intercept[RuntimeException] {
      val klayer = ZSqueeze[Float](-1, inputShape = Shape(2, 1, 3))
      klayer.build(Shape(-1, 2, 1, 3))
    }
  }

}

class SqueezeSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val squeeze = ZSqueeze[Float](dim = 2, inputShape = Shape(1, 1, 3))
    squeeze.build(Shape(2, 1, 1, 3))
    val input = Tensor[Float](2, 1, 1, 3).rand()
    runSerializationTest(squeeze, input)
  }
}
