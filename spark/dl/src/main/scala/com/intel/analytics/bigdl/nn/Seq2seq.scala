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
   shrinkHiddenStateModules: Array[Array[TensorModule[T]]] = null,
   decoderInputType: DecoderInputType = DecoderInputType.ENCODERINPUTLASTTIME)
  (implicit ev: TensorNumeric[T]) extends AbstractModule[Activity, Tensor[T], T] {
  var preDecoderInput: Tensor[T] = null
  var decoderInput: Tensor[T] = null
  var encoderInput: Tensor[T] = null
  var encoderOutput: Tensor[T] = null

  val encoder = if (preEncoder == null) Sequential()
    else Sequential().add(preEncoder)
  encoderRecs.foreach(encoder.add(_))

  val decoder = Sequential()
  decoderRecs.foreach(decoder.add(_))

  val module: AbstractModule[Activity, Activity, T] = if (preDecoder == null) {
    Sequential().add(encoder).add(decoder)
  } else {
    Sequential().add(encoder).add(preDecoder).add(decoder)
  }

  override def updateOutput(input: Activity): Tensor[T] = {
    val feedbackPreviousOutput = decoderRecs.head.isInstanceOf[RecurrentDecoder[T]]
    decoderInputType match {
      case DecoderInputType.ENCODERINPUTSPLIT =>
        encoderInput = input.toTable(1)
        preDecoderInput = input.toTable(2)
      case DecoderInputType.ENCODERINPUTLASTTIME =>
        require(feedbackPreviousOutput, "ENCODERINPUTLASTTIME can" +
          "only work with RecurrentDecoder")
        encoderInput = input.toTensor
        preDecoderInput = input.toTensor.select(2, input.toTensor.size(2))
      case _ => throw new IllegalArgumentException("Unknown decodeInput mode," +
        "current only support ENCODERINPUTSPLIT, ENCODERINPUTLASTTIME")
    }
    encoderOutput = encoder.forward(encoderInput).toTensor

    if (feedbackPreviousOutput) {
      if (encoderRecs.size == 1) {
        var newHiddenState = encoderRecs.head.getHiddenState()
        if (shrinkHiddenStateModules != null && shrinkHiddenStateModules.head != null) {
          newHiddenState = shrinkHiddenState(newHiddenState,
            shrinkHiddenStateModules.head)
        }
        decoderRecs.head.setHiddenState(newHiddenState)
      } else {
        val hiddenState = T()
        for((rec, i) <- encoderRecs.view.zipWithIndex) {
          if (shrinkHiddenStateModules != null &&
            shrinkHiddenStateModules(i) != null) {
            hiddenState(i) = shrinkHiddenState(rec.getHiddenState(),
              shrinkHiddenStateModules(i))
          } else hiddenState(i) = rec.getHiddenState()
        }
        decoderRecs.head.setHiddenState(hiddenState)
      }
    } else {
      for ((x, i) <- encoderRecs.view.zipWithIndex) {
        if (shrinkHiddenStateModules != null &&
          shrinkHiddenStateModules(i) != null) {
          decoderRecs(i).setHiddenState(
            shrinkHiddenState(x.getHiddenState(), shrinkHiddenStateModules(i)))
        } else decoderRecs(i).setHiddenState(x.getHiddenState())
      }
    }

    decoderInput = if (preDecoder != null) preDecoder.forward(preDecoderInput).toTensor
      else preDecoderInput
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
    val decoderGradInput = decoder.backward(decoderInput, gradOutput).toTensor
    if (preDecoder != null) {
      if (preDecoderInput.dim == decoderGradInput.dim) {
        preDecoder.backward(preDecoderInput, decoderGradInput)
      } else {
        preDecoder.backward(preDecoderInput, decoderGradInput.select(2, 1).contiguous())
      }
    }
    if (decoderRecs.head.isInstanceOf[RecurrentDecoder[T]]) {
      val gradHiddenStates = decoderRecs.head.getGradHiddenState()
      if (encoderRecs.size == 1) {
        var newGradHiddenStates = gradHiddenStates
        if (shrinkHiddenStateModules != null && shrinkHiddenStateModules.head != null) {
          newGradHiddenStates = shrinkGradHiddenState(encoderRecs.head.getHiddenState(),
            gradHiddenStates, shrinkHiddenStateModules.head)
        }
        encoderRecs.head.setGradHiddenState(newGradHiddenStates)
      } else {
          for ((x, i) <- encoderRecs.view.zipWithIndex) {
            if (shrinkHiddenStateModules != null &&
              shrinkHiddenStateModules(i) != null) {
              val newGradHiddenState = shrinkGradHiddenState(encoderRecs(i).getHiddenState(),
                gradHiddenStates.toTable(i), shrinkHiddenStateModules(i))
              x.setGradHiddenState(newGradHiddenState)
          } else x.setGradHiddenState(gradHiddenStates.toTable(i))
        }
      }
    } else {
      for ((x, i) <- decoderRecs.view.zipWithIndex) {
        var newGradHiddenState = x.getGradHiddenState()
        if (shrinkHiddenStateModules != null &&
        shrinkHiddenStateModules(i) != null) {
          newGradHiddenState = shrinkGradHiddenState(encoderRecs(i).getHiddenState(),
            newGradHiddenState, shrinkHiddenStateModules(i))
        }
        encoderRecs(i).setGradHiddenState(newGradHiddenState)
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
  
  private def shrinkHiddenState(state: Activity,
    shrinkModule: Array[TensorModule[T]]): Activity = {
    require(shrinkModule != null, "shrinkModule cannot be null")
    var newState: Activity = null
    if (state.isTensor) {
      require(shrinkModule.head != null, "shrinkModule element cannot be null")
      newState = shrinkModule.head.forward(state)
    } else {
      require(shrinkModule(0) != null && shrinkModule(1) != null,
        "shrinkModule element cannot be null")
      newState = T()
      newState.toTable(1) = shrinkModule(0).forward(state.toTable(1))
      newState.toTable(2) = shrinkModule(1).forward(state.toTable(2))
    }
    newState
  }

  private def shrinkGradHiddenState(state: Activity, gradState: Activity,
    shrinkModule: Array[TensorModule[T]]): Activity = {
    require(shrinkModule != null, "shrinkModule cannot be null")
    var newGradState: Activity = null
    if (gradState.isTensor) {
      require(shrinkModule(0) != null, "shrinkModule element cannot be null")
      newGradState = shrinkModule(0).backward(state, gradState)
    } else {
      require(shrinkModule(0) != null && shrinkModule(1) != null,
        "shrinkModule element cannot be null")
      newGradState = T()
      newGradState.toTable(1) = shrinkModule(0).backward(state.toTable(1),
        gradState.toTable(1))
      newGradState.toTable(2) = shrinkModule(1).backward(state.toTable(2),
        gradState.toTable(2))
    }
    newGradState
  }
}

object Seq2seq {
  def apply[@specialized(Float, Double) T: ClassTag](encoderCells: Array[Recurrent[T]],
     decoderCells: Array[Recurrent[T]], preEncoder: AbstractModule[Activity, Activity, T] = null,
     preDecoder: AbstractModule[Activity, Activity, T] = null,
     shrinkHiddenStateModules: Array[Array[TensorModule[T]]] = null,
     decoderInputType: DecoderInputType = DecoderInputType.ENCODERINPUTLASTTIME)
    (implicit ev: TensorNumeric[T]): Seq2seq[T] = {
    new Seq2seq[T](encoderCells, decoderCells, preEncoder, preDecoder,
      shrinkHiddenStateModules, decoderInputType)
  }
}

object DecoderInputType extends Enumeration {
  type DecoderInputType = Value
  val ENCODERINPUTSPLIT, ENCODERINPUTLASTTIME = Value
}
