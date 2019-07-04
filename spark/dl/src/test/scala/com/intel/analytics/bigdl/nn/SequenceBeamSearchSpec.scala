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
import com.intel.analytics.bigdl.utils.{T, Table}
import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

class SequenceBeamSearchSpec extends FlatSpec with Matchers{
  "beam search" should "work correctly" in {
    val batchSize = 2
    val beamSize = 3
    val vocabSize = 4
    val decodeLength = 10
    val alpha: Float = 0.0f
    val eosId = 1.0f
    val numHiddenLayers = 2
    val hiddenSize = 5
    val inputLength = 6

    def symbolsToLogitsFn(Ids: Tensor[Float], i: Int, maxDecoderLen: Int,
     encoder: Tensor[Float], Bias: Tensor[Float], list1: List[Tensor[Float]],
     list2: List[Tensor[Float]]):
    (Tensor[Float], Tensor[Float], Tensor[Float], List[Tensor[Float]], List[Tensor[Float]]) = {
      val tensor = Tensor(Array(0.14f, 0.62f, 0.02f, 0.93f,
        0.59f, 0.48f, 0.27f, 0.70f,
        0.11f, 0.30f, 0.35f, 0.15f,
        0.67f, 0.39f, 0.33f, 0.01f,
        0.44f, 0.52f, 0.45f, 0.23f,
        0.75f, 0.79f, 0.26f, 0.47f), Array(6, 4))
      val encoder1 = encoder + Tensor[Float](encoder.size()).rand()
      val Bias1 = Bias + Tensor[Float](Bias.size()).rand()
      val batch_beam = encoder.size()(0)
      list1.map(e => Tensor[Float](batch_beam, 1, hiddenSize).rand())
      list2.map(e => Tensor[Float](batch_beam, 1, hiddenSize).rand())
      (tensor, encoder1, Bias1, list1, list2)
    }

    val beamSearch = new SequenceBeamSearch[Float](vocabSize,
      beamSize, alpha, decodeLength, eosId, numHiddenLayers, hiddenSize)
    beamSearch.setLogitFn(symbolsToLogitsFn)
    val encodeOutputs = Tensor[Float](batchSize, inputLength, hiddenSize).rand()
    val encoderDecoderAttentionBias = Tensor[Float](batchSize, 1, 1, inputLength).rand()
    val output = beamSearch.forward(T(encodeOutputs, encoderDecoderAttentionBias))
      .asInstanceOf[Table]
    val outputSeq = Tensor[Float].resizeAs(output[Tensor[Float]](1)).copy(output[Tensor[Float]](1))
    val outputScore = Tensor[Float].resizeAs(output[Tensor[Float]](2))
     .copy(output[Tensor[Float]](2))
    beamSearch.clearState()
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

class SequenceBeamSearchSerialTest extends ModuleSerializationTest{
  override def test(): Unit = {
  // skip serial test
  }
}
