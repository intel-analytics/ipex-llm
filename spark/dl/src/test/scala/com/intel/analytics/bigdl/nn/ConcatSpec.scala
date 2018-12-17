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
import com.intel.analytics.bigdl.utils.LayerException
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

@com.intel.analytics.bigdl.tags.Parallel
class ConcatSpec extends FlatSpec with Matchers {

  "toString" should "return good value" in {
    val seq1 = new Sequential[Double]
    seq1.add(new Linear(10, 15))
    seq1.add(new Sigmoid)

    val seq2 = new Sequential[Double]
    seq2.add(new Linear(10, 15))
    seq2.add(new Tanh)

    val concat = new Concat[Double](2)
    concat.add(seq1)
    concat.add(seq2)

    println(concat)

  }

  "Concat forward/backward 1D input/output" should "return good result" in {
    val model = Concat[Float](1)
    model.add(Identity[Float]())
    model.add(Identity[Float]())
    val input = Tensor[Float].range(1, 3, 1)
    val gradOutput = Tensor[Float].range(1, 6, 1)
    val output = model.forward(input)
    val gradInput = model.backward(input, gradOutput)
    output should be (Tensor(Storage(Array[Float](1, 2, 3, 1, 2, 3))))
    gradInput should be (Tensor(Storage(Array[Float](5, 7, 9))))
  }

  "Concat forward/backward 4D input/output" should "return good result" in {
    val model = Concat[Float](3)
    model.add(Identity[Float]())
    model.add(AddConstant[Float](1))
    val input = Tensor[Float](2, 2, 2, 2).apply1(_ => 1)
    var i = 0
    val gradOutput = Tensor[Float](2, 2, 4, 2).apply1 { _ =>
      val result = if (i % 8 < 4) 2f else 3f
      i = i + 1
      result
    }
    val output = model.forward(input)
    val expectedOutput = Tensor[Float](2, 2, 4, 2).apply1 { _ =>
      val result = if (i % 8 < 4) 1f else 2f
      i = i + 1
      result
    }
    val gradInput = model.backward(input, gradOutput)
    val expectedGradInput = Tensor[Float](2, 2, 2, 2).apply1(_ => 5f)
    output should be (expectedOutput)
    gradInput should be (expectedGradInput)
  }

  "Concat with incorrec input" should "throw expected exception" in {
    val model = Concat[Float](2)
    model.add(Reshape[Float](Array(5, 2)))
    model.add(Reshape[Float](Array(2, 5)))
    val input = Tensor[Float](10)
    val caught = intercept[LayerException] {
      model.forward(input)
    }
    val contains = caught.error.getMessage.contains("output size at dimension 1 mismatch")
    contains should be (true)
  }
}

class ConcatSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val input = Tensor[Float](2, 2, 2).apply1(e => Random.nextFloat())
    val concat = Concat[Float](2).setName("concat")
    concat.add(Abs[Float]())
    concat.add(Abs[Float]())
    runSerializationTest(concat, input)
  }
}
