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

import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

object BeamSearchTest {
  def main(args: Array[String]): Unit = {
    val batch_size = 2
    val beam_size = 3
    val vocab_size = 4
    val decode_length = 10
    val alpha: Float = 0.0f
    val eosId = 1.0f

    def symbolsToLogitsFn(Ids: Tensor[Float], i: Tensor[Int],
      cache1: Map[String, Map[String, Tensor[Float]]], cache2: Map[String, Tensor[Float]]):
      (Tensor[Float], Map[String, Map[String, Tensor[Float]]], Map[String, Tensor[Float]]) = {
      //      val tensor = Tensor(batch_size*beam_size, vocab_size)
      //        .apply1(e => Random.nextFloat())
      val tensor = Tensor(Array(0.14f, 0.62f, 0.02f, 0.93f,
        0.59f, 0.48f, 0.27f, 0.70f,
        0.11f, 0.30f, 0.35f, 0.15f,
        0.67f, 0.39f, 0.33f, 0.01f,
        0.44f, 0.52f, 0.45f, 0.23f,
        0.75f, 0.79f, 0.26f, 0.47f), Array(batch_size*beam_size, vocab_size))
      println("symbols", tensor)
      (tensor, cache1, cache2)
    }

    val BeamSearch = SequenceBeamSearch(symbolsToLogitsFn, vocab_size, batch_size,
      beam_size, alpha, decode_length, eosId)
    val initialId = Tensor(Array(0.0f, 0.0f), Array(2))
    val tensor1 = Tensor(Array(1.0f, 2, 3, 1, 2, 3), Array(2, 3))
    val tensor2 = Tensor(Array(1.0f, 2, 3, 1, 2, 3), Array(2, 3))
    // println(tensor1.cmul(tensor1))
    // val outputs = BeamSearch.logProbFromLogits(tensor)
    val initialCache = Map("a" -> Map("i" -> tensor1, "j" -> tensor2))
    val initialCache2 = Map("a" -> tensor1)
    BeamSearch.search(initialId, initialCache, initialCache2)
  }

}
