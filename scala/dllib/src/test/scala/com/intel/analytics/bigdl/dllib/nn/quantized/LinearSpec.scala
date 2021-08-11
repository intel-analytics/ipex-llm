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

package com.intel.analytics.bigdl.nn.quantized

import com.intel.analytics.bigdl.nn.{Module, Linear => NNLinear}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers, ParallelTestExecution}

import scala.util.Random

@com.intel.analytics.bigdl.tags.Parallel
class LinearSpec extends FlatSpec with Matchers with ParallelTestExecution {
  val testCases = List(
    TestCase(1, 1, 1),
    TestCase(3, 3, 4),
    TestCase(512, 1024, 32),
    TestCase(4, 2302, 1152)
  )

  for (test <- testCases) {
    val start = s"A $test"
    start should "generate correct output compared with nn.Linear" in {
      val weight = Tensor(test.outputSize, test.inputSize).fill(1.0f)
      val bias = Tensor(test.outputSize).fill(0f)
      val input = Tensor(test.batchSize, test.inputSize).fill(1.0f)

      val nnLinear = NNLinear(test.inputSize, test.outputSize, initWeight = weight, initBias = bias)
      val quantizedLinear = Linear(test.inputSize, test.outputSize, initWeight = weight,
        initBias = bias)

      nnLinear.updateOutput(input)
      quantizedLinear.updateOutput(input)

      nnLinear.output shouldEqual quantizedLinear.output

      quantizedLinear.release()
    }
  }

  case class TestCase(batchSize: Int, inputSize: Int, outputSize: Int)
}

class LinearSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val linear = NNLinear[Float](10, 2).setName("linear")
    val input = Tensor[Float](10).apply1(_ => Random.nextFloat())
    runSerializationTest(linear, input)
  }
}
