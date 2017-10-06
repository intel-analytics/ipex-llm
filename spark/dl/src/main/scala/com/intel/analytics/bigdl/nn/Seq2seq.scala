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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.nn.DecoderInputType.DecoderInputType
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag
import com.intel.analytics.bigdl.utils.{T, Table}

class Seq2seq[T: ClassTag](encoderRecs: Array[Recurrent[T]], decoderRecs: Array[Recurrent[T]],
  preEncoder: AbstractModule[Activity, Activity, T] = null,
  preDecoder: AbstractModule[Activity, Activity, T] = null,
  decoderInputType: DecoderInputType = DecoderInputType.ENCODERINPUTLASTTIME)
  (implicit ev: TensorNumeric[T]) extends AbstractModule[Activity, Tensor[T], T] {
  var decoderInput: Tensor[T] = null
  var encoderInput: Tensor[T] = null
  var encoderOutput: Tensor[T] = null

  val encoder = if (preDecoder == null) Sequential() else Sequential().add(preEncoder)
//  for (i <- 0 until encoderCells.size) {
//    val rec = Recurrent[T]().add(encoderCells(i))
//    encoderRecs(i) = rec
//    encoder.add(rec)
//  }
  encoderRecs.foreach(encoder.add(_))

  val decoder = if (preDecoder == null) Sequential() else Sequential().add(preDecoder)
//  if (feedPreviousOutput) {
//    require(outputLength >= 1, "output length must be greater than 0")
//    decoderRecs(0) = if (decoderCells.length == 1) {
//      RecurrentDecoder(outputLength).add(decoderCells.head)
//    } else RecurrentDecoder(outputLength).add(MultiRNNCell(decoderCells))
//    decoder.add(decoderRecs.head)
//  } else {
//    for ((x, i) <- decoderCells.view.zipWithIndex) {
//      decoderRecs(i) = Recurrent[T]().add(x)
//      decoder.add(decoderRecs(i))
//    }
//  }
  decoderRecs.foreach(decoder.add(_))

  val module: AbstractModule[Activity, Activity, T] = Sequential().add(encoder).add(decoder)

  override def updateOutput(input: Activity): Tensor[T] = {
    val feedbackPreviousOutput = decoderRecs.head.isInstanceOf[RecurrentDecoder[T]]
    decoderInputType match {
      case DecoderInputType.ENCODERINPUTSPLIT =>
        encoderInput = input.toTable[Tensor[T]](1)
        decoderInput = input.toTable[Tensor[T]](2)
      case DecoderInputType.ENCODERINPUTLASTTIME =>
        require(feedbackPreviousOutput, "ENCODERINPUTLASTTIME can" +
          "only work with RecurrentDecoder")
        encoderInput = input.toTensor
        decoderInput = input.toTensor.select(2, input.toTensor.size(2))
      case DecoderInputType.ZEROS =>
        encoderInput = input.toTensor
        decoderInput = null
      case _ => throw new IllegalArgumentException("Unknown decodeInput mode")
    }
    encoderOutput = encoder.forward(encoderInput).toTensor
    if (decoderInput == null) {
      decoderInput = if (feedbackPreviousOutput) {
        val _outputSize = encoderOutput.size()
        Tensor[T](Array(_outputSize(0)) ++ _outputSize.drop(2))
      } else Tensor[T](encoderOutput.size())
    }

    if (feedbackPreviousOutput) {
      if (encoderRecs.size == 1) {
        decoderRecs.head.setHiddenState(encoderRecs.head.getHiddenState())
      } else {
        val hiddenState = T()
        for((rec, i) <- encoderRecs.view.zipWithIndex) {
          hiddenState(i) = rec.getHiddenState()
        }
        decoderRecs.head.setHiddenState(hiddenState)
      }
    } else {
      for ((x, i) <- encoderRecs.view.zipWithIndex) {
        decoderRecs(i).setHiddenState(x.getHiddenState())
      }
    }

    output = decoder.forward(decoderInput).toTensor
    output
  }

  override def accGradParameters(input: Activity, gradOutput: Tensor[T]): Unit = {
    throw new Exception("Should not enter seq2seq accGradParameters" +
      "as it has override backward")
  }

  override def updateGradInput(input: Activity, gradOutput: Tensor[T]): Tensor[T] = {
    throw new Exception("Should not enter seq2seq updateGradInput" +
      "as it has override backward")
    gradInput.toTensor
  }

  override def backward(input: Activity, gradOutput: Tensor[T]): Tensor[T] = {
    decoder.backward(decoderInput, gradOutput)
    if (decoderRecs.head.isInstanceOf[RecurrentDecoder[T]]) {
      val gradHiddenStates = decoderRecs.head
        .asInstanceOf[RecurrentDecoder[T]].getGradHiddenState()
      if (encoderRecs.size == 1) {
        encoderRecs.head.setGradHiddenState(gradHiddenStates)
      } else {
        for ((x, i) <- encoderRecs.view.zipWithIndex) {
          x.setGradHiddenState(gradHiddenStates.toTable(i))
        }
      }
    } else {
      for ((x, i) <- decoderRecs.view.zipWithIndex) {
        encoderRecs(i).setGradHiddenState(x.getGradHiddenState())
      }
    }

    gradInput = encoder.backward(encoderInput, Tensor[T](encoderOutput.size())).toTensor

    gradInput.toTensor
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    module.parameters()
  }

  override def getParametersTable(): Table = {
    module.getParametersTable()
  }
}

object Seq2seq {
  def apply[@specialized(Float, Double) T: ClassTag](encoderCells: Array[Recurrent[T]],
    decoderCells: Array[Recurrent[T]], preEncoder: AbstractModule[Activity, Activity, T] = null,
    preDecoder: AbstractModule[Activity, Activity, T] = null,
    decoderInputType: DecoderInputType = DecoderInputType.ENCODERINPUTLASTTIME)
    (implicit ev: TensorNumeric[T]): Seq2seq[T] = {
    new Seq2seq[T](encoderCells, decoderCells, preEncoder, preDecoder, decoderInputType)
  }
}

object DecoderInputType extends Enumeration {
  type DecoderInputType = Value
  val ZEROS, ENCODERINPUTSPLIT, ENCODERINPUTLASTTIME = Value
}
