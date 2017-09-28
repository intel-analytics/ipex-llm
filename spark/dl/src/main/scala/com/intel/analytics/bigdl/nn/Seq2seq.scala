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

import com.intel.analytics.bigdl.nn.abstractnn.{Activity, TensorModule}
import com.intel.analytics.bigdl.nn.DecoderInputType.DecoderInputType
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class Seq2seq[T: ClassTag](encoderCells: Array[Cell[T]], decoderCells: Array[Cell[T]],
  outputLength: Int, broadcastState: Boolean = false,
  decoderInputType: DecoderInputType = DecoderInputType.ENCODERINPUT)
  (implicit ev: TensorNumeric[T]) extends TensorModule[T] {
  var states: Array[Activity] = null
  var recs = new Array[Recurrent[T]](encoderCells.length)
  var decoderInput: Tensor[T] = null
  var encoderOutput: Tensor[T] = null
    
  val encoder = Sequential()
  for (i <- 0 until encoderCells.size) {
    val rec = Recurrent[T]()
    rec.add(encoderCells(i))
    recs(i) = rec
    encoder.add(rec)
  }
   
  val decoder = if (decoderCells.length == 1) RecurrentDecoder(outputLength).add(decoderCells.head)
    else RecurrentDecoder(outputLength).add(MultiCell(decoderCells))
  
  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim == 3 || input.dim == 5 || input.dim == 6,
      "Recurrent: input should be a 3D/5D/6D Tensor, e.g [batch, times, nDim], " +
        s"current input.dim = ${input.dim}")
    
    encoderOutput = encoder.forward(input).toTensor
    if (broadcastState) {
      require(encoderCells.length == decoderCells.length)
      states = new Array[Activity](encoderCells.length)
      var i = 0
      while (i < encoderCells.length) {
        states(i) = recs(i).getState()
        i += 1
      }
      if (decoderCells.length == 1) {
        decoder.setState(states(i - 1))
      } else {
        decoder.setStates(states)
      }
    }

    val _outputSize = encoderOutput.size()
    decoderInputType match {
      case DecoderInputType.ENCODEROUTPUT =>
        decoderInput = encoderOutput.select(2, encoderOutput.size(2))
      case DecoderInputType.ENCODERINPUT => decoderInput = input.select(2, input.size(2))
      case DecoderInputType.ZEROS => decoderInput =
        Tensor[T](Array(_outputSize(0)) ++ _outputSize.drop(2))
      case _ => throw new IllegalArgumentException("Unknown decodeInput mode")
    }
    
    output = decoder.forward(decoderInput)
    output
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    throw new Exception("Should not enter seq2seq accGradParameters" +
      "as it has override backward")
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    throw new Exception("Should not enter seq2seq updateGradInput" +
      "as it has override backward")
    gradInput
  }

  override def backward(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val gradInputDecoder = decoder.backward(decoderInput, gradOutput)
    if (broadcastState) {
      if (decoderCells.length == 1) {
        val gradState = decoder.getGradState()
        recs(encoderCells.length - 1).setGradState(gradState)
      } else {
        val gradStates = decoder.getGradStates()
        var i = 0
        while ( i < gradStates.length) {
          recs(i).setGradState(gradStates(i))
          i += 1
        }
      }
    }

    decoderInputType match {
      case DecoderInputType.ENCODEROUTPUT =>
        encoder.backward(input, gradInputDecoder).toTensor
      case DecoderInputType.ENCODERINPUT => encoder.backward(input,
        Tensor[T](encoderOutput.size())).toTensor
      case DecoderInputType.ZEROS => gradInput = encoder.backward(input,
        Tensor[T](encoderOutput.size())).toTensor
        
      case _ => throw new IllegalArgumentException("Unknown decodeInput mode")
    }
    
    gradInput
  }
}

object Seq2seq {
  def apply[@specialized(Float, Double) T: ClassTag](encoderCells: Array[Cell[T]],
    decoderCells: Array[Cell[T]], outputLength: Int, broadcastState: Boolean = false,
    decoderInput: DecoderInputType = DecoderInputType.ENCODERINPUT)
    (implicit ev: TensorNumeric[T]): Seq2seq[T] = {
    new Seq2seq[T](encoderCells, decoderCells, outputLength, broadcastState, decoderInput)
  }
}

object DecoderInputType extends Enumeration {
  type DecoderInputType = Value
  val ZEROS, ENCODEROUTPUT, ENCODERINPUT = Value
}
