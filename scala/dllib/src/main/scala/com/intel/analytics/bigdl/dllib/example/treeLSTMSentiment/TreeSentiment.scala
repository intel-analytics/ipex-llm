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

package com.intel.analytics.bigdl.example.treeLSTMSentiment

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor

import scala.language.existentials

object TreeLSTMSentiment {
  def apply(
    word2VecTensor: Tensor[Float],
    hiddenSize: Int,
    classNum: Int,
    p: Double = 0.5
  ): Module[Float] = {
    val vocabSize = word2VecTensor.size(1)
    val embeddingDim = word2VecTensor.size(2)
    val embedding = LookupTable(vocabSize, embeddingDim)
    embedding.weight.set(word2VecTensor)
    embedding.setScaleW(2)

    val treeLSTMModule = Sequential()
      .add(BinaryTreeLSTM(
        embeddingDim, hiddenSize, withGraph = true))
      .add(TimeDistributed(Dropout(p)))
      .add(TimeDistributed(Linear(hiddenSize, classNum)))
      .add(TimeDistributed(LogSoftMax()))

    Sequential()
      .add(MapTable(Squeeze(3)))
      .add(ParallelTable()
        .add(embedding)
        .add(Identity()))
      .add(treeLSTMModule)
  }
}
