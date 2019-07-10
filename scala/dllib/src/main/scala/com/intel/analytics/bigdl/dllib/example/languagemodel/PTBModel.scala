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

package com.intel.analytics.bigdl.example.languagemodel

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn.{TimeDistributed, _}
import com.intel.analytics.bigdl.utils.{Engine, MklDnn}

object PTBModel {
  def transformer(
     inputSize: Int = 10000,
     hiddenSize: Int = 256,
     outputSize: Int = 10000,
     numLayers: Int = 2,
     keepProb: Float = 2.0f)
  : Module[Float] = {
    val input = Input[Float]()
    val transformer = Transformer[Float](vocabSize = inputSize,
      hiddenSize = hiddenSize, numHeads = 4, filterSize = hiddenSize*4,
      numHiddenlayers = numLayers, embeddingDropout = 1- keepProb,
      attentionDropout = 0.1f, ffnDropout = 0.1f).inputs(input)
    val linear = Linear[Float](hiddenSize, outputSize)
    val output = TimeDistributed[Float](linear).inputs(transformer)
    Graph(input, output)
  }

  def lstm(
    inputSize: Int,
    hiddenSize: Int,
    outputSize: Int,
    numLayers: Int,
    keepProb: Float = 2.0f)
  : Module[Float] = {
    val input = Input[Float]()
    val embeddingLookup = LookupTable[Float](inputSize, hiddenSize).inputs(input)

    val inputs = if (keepProb < 1) {
      Dropout[Float](keepProb).inputs(embeddingLookup)
    } else embeddingLookup

    val lstm = addLayer(hiddenSize, hiddenSize, 1, numLayers, inputs)
    val linear = Linear[Float](hiddenSize, outputSize)
    val output = TimeDistributed[Float](linear).inputs(lstm)

    val model = Graph(input, output)
    model.asInstanceOf[StaticGraph[Float]].setInputFormats(Seq(Memory.Format.nc))
    model.asInstanceOf[StaticGraph[Float]].setOutputFormats(Seq(Memory.Format.ntc))
    if (Engine.getEngineType() == MklDnn) model.asInstanceOf[StaticGraph[Float]].toIRgraph()
    else model
  }

  private def addLayer(inputSize: Int,
               hiddenSize: Int,
               depth: Int,
               numLayers: Int,
               input: ModuleNode[Float]): ModuleNode[Float] = {
    if (depth == numLayers) {
      Recurrent[Float]()
        .add(LSTM[Float](inputSize, hiddenSize, 0, null, null, null))
        .inputs(input)
    } else {
      addLayer(
        inputSize,
        hiddenSize,
        depth + 1,
        numLayers,
        Recurrent[Float]()
          .add(LSTM[Float](inputSize, hiddenSize, 0, null, null, null))
          .inputs(input)
      )
    }
  }
}
