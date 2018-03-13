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
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class SplitTableSpec extends FlatSpec with BeforeAndAfter with Matchers {

  "A BifurcateSplitTable " should "generate correct output and grad" in {
    val seed = 100
    Random.setSeed(seed)

    val dim = 2
    val module = new BifurcateSplitTable[Double](dim)
    val input = Tensor[Double](3, 4).randn()
    val expectedGradInput = Tensor[Double]().resizeAs(input).randn()
    val gradOutput = T(expectedGradInput.narrow(dim, 1, 2), expectedGradInput.narrow(dim, 3, 2))

    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)

    output.length() should be (2)
    val left = output(1).asInstanceOf[Tensor[Double]]
    val right = output(2).asInstanceOf[Tensor[Double]]
    left should be (input.narrow(dim, 1, 2))
    right should be (input.narrow(dim, 3, 2))

    gradInput should be (expectedGradInput)
  }
}

class BifurcateSplitTableSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val batchNorm = BifurcateSplitTable[Float](1).setName("batchNorm")
    val input = Tensor[Float](2, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(batchNorm, input)
  }
}

class SplitTableSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val splitTable = SplitTable[Float](2).setName("splitTable")
    val input = Tensor[Float](2, 10).apply1( e => Random.nextFloat())
    runSerializationTest(splitTable, input)
  }
}
