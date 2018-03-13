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
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

  /**
  * Created by yao on 9/20/16.
  */
@com.intel.analytics.bigdl.tags.Parallel
class IdentitySpec extends FlatSpec with Matchers {
  "Identity" should "generate correct output and grad" in {
    val batchN = 3
    val inputN = 5
    val outputN = inputN

    val input = Tensor[Double](batchN, inputN)
    input.rand()
    val gradOutput = Tensor[Double](batchN, outputN)
    gradOutput.rand()

    val module = new Identity[Double]()
    val output = module.forward(input)
    assert(input equals output)

    val gradInput = module.backward(input, gradOutput)
    assert(gradInput equals gradOutput)
  }
}

class IdentitySerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val identity = Identity[Float]().setName("identity")
    val input = Tensor[Float](10).apply1(_ => Random.nextFloat())
    runSerializationTest(identity, input)
  }
}
