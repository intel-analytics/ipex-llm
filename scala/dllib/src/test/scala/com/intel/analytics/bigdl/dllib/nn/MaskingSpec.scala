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

import com.intel.analytics.bigdl.keras.KerasBaseSpec
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

import scala.util.Random

class MaskingSpec extends KerasBaseSpec {

  "Masking" should "generate corrent result when batchsize == 1" in {
    val inputSize = 2
    val times = 7
    val batchSize = 1
    val mask_value = -1

    val sigmoidCode =
      s"""
         |input_tensor = Input(shape=[${times}, ${inputSize}])
         |input = np.array([1, 1, ${mask_value}, 2, 3, 3, 4, 4, ${mask_value}, ${mask_value}, 6, 6,
         |7, 7]).reshape(${batchSize}, ${times}, ${inputSize})
         |output_tensor = Masking(${mask_value})(input_tensor)
         |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin

    val masking = Masking[Float](mask_value)

    checkOutputAndGrad(masking, sigmoidCode)
  }

  "Masking" should "generate corrent result when batchsize != 1" in {
    val inputSize = 2
    val times = 7
    val batchSize = 3
    val mask_value = -1

    val sigmoidCode =
      s"""
        |input_tensor = Input(shape=[${times}, ${inputSize}])
        |input = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, ${mask_value}, 1, 2,
        |        2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 1, 1, 2, 2, 3, 3,
        |        ${mask_value}, ${mask_value}, 5,
        |        5, 6, 6, 7, 7]).reshape(${batchSize}, ${times}, ${inputSize})
        |output_tensor = Masking(${mask_value})(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin

    val masking = Masking[Float](mask_value)

    checkOutputAndGrad(masking, sigmoidCode)
  }
}


class MaskingSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val masking = Masking[Float](0.1).setName("masking")
    val input = Tensor[Float](2, 3, 4).apply1(_ => Random.nextFloat())
    runSerializationTest(masking, input)
  }
}
