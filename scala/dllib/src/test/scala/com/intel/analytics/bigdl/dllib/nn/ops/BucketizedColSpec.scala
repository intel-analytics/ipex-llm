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

class BucketizedColSpec
  extends FlatSpec with Matchers {
  "BucketizedCol with Double Type" should "work correctly" in {
    val input = Tensor[Double](T(T(-1, 1), T(101, 10), T(5, 100)))
    val expectOutput = Tensor[Int](
      T(T(0, 1), T(3, 2), T(1, 3))
    )
    val output = BucketizedCol[Double](boundaries = Array(0, 10, 100))
      .forward(input)
    output should be(expectOutput)
  }

  "BucketizedCol with Float Type" should "work correctly" in {
    val input = Tensor[Float](T(T(-1.0f, 1.0f), T(101.0f, 10.0f), T(5.0f, 100.0f)))
    val expectOutput = Tensor[Int](
      T(T(0, 1), T(3, 2), T(1, 3))
    )
    val output = BucketizedCol[Float](boundaries = Array(0, 10, 100))
      .forward(input)
    output should be(expectOutput)
  }
}

class BucketizedColSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val bucketizedCol = BucketizedCol[Float](boundaries = Array(0.0, 10.0, 100.0))
      .setName("bucketizedCol")
    val input = Tensor[Float](T(T(-1, 1), T(101, 10), T(5, 100)))
    runSerializationTest(bucketizedCol, input)
  }
}
