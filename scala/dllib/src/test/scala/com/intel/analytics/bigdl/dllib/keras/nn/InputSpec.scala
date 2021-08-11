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

package com.intel.analytics.bigdl.keras.nn

import com.intel.analytics.bigdl.nn.keras.Merge.merge
import com.intel.analytics.bigdl.nn.keras._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{BigDLSpecHelper, Shape}
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import com.intel.analytics.bigdl.numeric.NumericFloat

import scala.util.Random

class InputSpec extends BigDLSpecHelper {
  "Duplicate container" should "not throw exception" in {
    val bx1 = Input(Shape(4))
    val bx2 = Input(Shape(5))
    val by1 = Dense(6, activation = "sigmoid").inputs(bx1)
    val bbranch1 = Model(bx1, by1).inputs(bx1)
    val bbranch2 = Dense(8).inputs(bx2)
    val bz = merge(List(bbranch1, bbranch2), mode = "concat")
    val bmodel = Model(Array(bx1, bx2), bz)

    // No exception should be threw in the above code
  }

  "Duplicate input layer" should "throw exception" in {
    val i = InputLayer(Shape(4))
    val seq = Sequential()
    seq.add(i)

    intercept[IllegalArgumentException] {
      seq.add(i)
    }
  }
}

class InputSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val input = InputLayer[Float](inputShape = Shape(20))
    val seq = Sequential[Float]()
    seq.add(input)
    val inputData = Tensor[Float](2, 20).apply1(_ => Random.nextFloat())
    runSerializationTest(seq, inputData)
  }
}
