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
import com.intel.analytics.bigdl.utils.{T, Table}
import org.scalatest.{FlatSpec, Matchers}



class SequenceBeamSearchSpec extends FlatSpec with Matchers{
  "beam search" should "work correctly" in {
    val batch_size = 2
    val beam_size = 3
    val vocab_size = 4
    val decode_length = 10
    val alpha: Float = 0.0f
    val eosId = 1.0f
    val numHiddenLayers = 2
    val hiddenSize = 5
    val inputlength = 6
    val beamSearch = new SequenceBeamSearch[Float](vocab_size, batch_size,
     beam_size, alpha, decode_length, eosId, numHiddenLayers, hiddenSize)
    val output = beamSearch.forward(T()).asInstanceOf[Table]
    val outputSeq = output[Tensor[Float]](1)
    val outputScore = output[Tensor[Float]](2)
    val expectedOutputSeq = Tensor[Float](
      T(T(T(0.0, 1.0, 0.0, 0.0, 0.0),
        T(0.0, 3.0, 1.0, 0.0, 0.0),
        T(0.0, 3.0, 3.0, 1.0, 0.0)),
        T(T(0.0, 1.0, 0.0, 0.0, 0.0),
        T(0.0, 0.0, 1.0, 0.0, 0.0),
        T(0.0, 2.0, 1.0, 0.0, 0.0))))
    val expectedOutputScore = Tensor[Float](
      T(T(-1.2615868, -2.2131736, -3.1647604),
        T(-1.3734006, -2.4668012, -2.715382)))
    outputSeq should be(expectedOutputSeq)
    outputScore should be(expectedOutputScore)

  }
}
