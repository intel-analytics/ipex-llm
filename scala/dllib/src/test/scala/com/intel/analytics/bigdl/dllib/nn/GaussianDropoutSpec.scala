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
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

  /**
  * Unit test for GaussianDropout.
  */
@com.intel.analytics.bigdl.tags.Parallel
class GaussianDropoutSpec extends FlatSpec with Matchers {
  "GaussianDropout" should "run through without problem in training mode" in {
    val batchN = 3
    val inputN = 5
    val outputN = inputN

    val input = Tensor[Double](batchN, inputN).rand()
    val gradOutput = Tensor[Double](batchN, outputN).rand()


    val module = new GaussianDropout[Double](0.5)
    // training mode
    module.training()
    val output = module.forward(input)
    // check size, output should be same size as input
    assertIntArrayEqual(output.size(), input.size())

    val gradInput = module.backward(input, gradOutput)
    assertIntArrayEqual(gradInput.size(), gradOutput.size())
  }

  "GaussianDropout" should "run correctly in evaluation mode" in {
    val batchN = 3
    val inputN = 5
    val outputN = inputN

    val input = Tensor[Double](batchN, inputN).rand()
    val gradOutput = Tensor[Double](batchN, outputN).rand()

    val module = new GaussianDropout[Double](0.5)
    module.evaluate()
    val outputEval = module.forward(input)
    // output should be the same as input
    assert(input equals outputEval)
    // backward reports error in evaluation mode
    intercept[IllegalArgumentException] {
      module.backward(input, gradOutput)
    }

  }

  "GaussianDropout" should "throw exception for illegal rate argument" in {
    intercept[IllegalArgumentException] {
      val module = new GaussianDropout[Double](-0.1)
    }
    intercept[IllegalArgumentException] {
      val module = new GaussianDropout[Double](2)
    }

  }

  def assertIntArrayEqual(a1: Array[Int], a2: Array[Int]): Unit = {
    (a1 zip a2).foreach(x => assert(x._1 == x._2))
  }
}

class GaussianDropoutSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    RNG.setSeed(1000)
    val gaussianDropout = GaussianDropout[Float](0.5).setName("gaussianDropout")
    val input = Tensor[Float](10).apply1(_ => Random.nextFloat())
    runSerializationTest(gaussianDropout, input)
  }
}
