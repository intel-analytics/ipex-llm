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

package com.intel.analytics.bigdl.nn.mkldnn

import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.nn.mkldnn.Phase.{InferencePhase, TrainingPhase}
import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}

class DropoutSpec extends FlatSpec with Matchers {
  "dropout output and gradinput" should "work correctly" in {
    val input = Tensor[Float](Array(2, 3, 4, 4)).fill(1)
    val zeros = Tensor[Float](Array(2, 3, 4, 4)).fill(0)

    val dropout = Dropout()
    dropout.setRuntime(new MklDnnRuntime)
    dropout.initFwdPrimitives(Array(HeapData(Array(2, 3, 4, 4), Memory.Format.nchw)), TrainingPhase)

    {
      dropout.forward(input)

      val notEqZeros = dropout.output.toTensor[Float].storage().array().count(_ != 0)
      val total = input.nElement()
      val ratio = notEqZeros.toDouble / total
      ratio should not be (1.0)
      ratio should not be (0.0)
    }

    {
      dropout.backward(input, dropout.output)
      val notEqZeros = dropout.gradInput.toTensor[Float].storage().array().count(_ != 0)
      val total = input.nElement()
      val ratio = notEqZeros.toDouble / total
      ratio should not be (1.0)
      ratio should not be (0.0)
    }
  }

  "dropout infer" should "work correctly" in {
    val input = Tensor[Float](Array(2, 3, 4, 4)).fill(1)
    val zeros = Tensor[Float](Array(2, 3, 4, 4)).fill(0)

    val dropout = Dropout()
    dropout.setRuntime(new MklDnnRuntime)
    dropout.initFwdPrimitives(Array(HeapData(Array(2, 3, 4, 4), Memory.Format.nchw)),
      InferencePhase)
    dropout.evaluate()

    dropout.forward(input)

    val notEqZeros = dropout.output.toTensor[Float].storage().array().count(_ != 0)
    val total = input.nElement()
    val ratio = notEqZeros.toDouble / total
    ratio should be (1.0)
  }

  "dropout in sequential" should "work correctly" in {
    val shape = Array(2, 3, 4, 4)
    val dropout = Dropout()
    val seq = Sequential().add(Input(shape, Memory.Format.nchw))
      .add(dropout)
      .add(Output(Memory.Format.nchw))

    seq.compile(TrainingPhase)

    val input = Tensor[Float](shape).rand(-1, 1)
    seq.forward(input)
    seq.backward(input, seq.output)
  }
}
