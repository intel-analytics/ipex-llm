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

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

@com.intel.analytics.bigdl.tags.Parallel
class LogSpec extends FlatSpec with Matchers {
  "A Log" should "generate correct output" in {
    val input = Tensor(Storage[Double](Array(1.0, 2, 3, 4, 5, 6)), 1, Array(2, 3))

    val output = Tensor(Storage(Array(0.0, 0.6931471805599453, 1.0986122886681098,
      1.3862943611198906, 1.6094379124341003, 1.791759469228055)), 1, Array(2, 3))

    val log = new Log[Double]()

    val logOutput = log.forward(input)

    logOutput should equal (output)
  }

  "A Log" should "generate correct grad" in {
    val input = Tensor(Storage[Double](Array(1.0, 2, 3, 4, 5, 6)), 1, Array(2, 3))

    val gradOutput = Tensor(Storage(Array(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)), 1, Array(2, 3))

    val log = new Log[Double]()

    val gradInput = log.backward(input, gradOutput)

    gradInput should equal (Tensor(Storage(Array(0.1, 0.1, 0.1, 0.1, 0.1, 0.1)), 1, Array(2, 3)))
  }
}

class LogSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val log = Log[Float]().setName("log")
    val input = Tensor[Float](10).apply1(_ => Random.nextFloat())
    runSerializationTest(log, input)
  }
}
