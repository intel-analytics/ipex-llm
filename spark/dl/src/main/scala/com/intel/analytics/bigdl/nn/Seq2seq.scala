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
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import com.intel.analytics.bigdl.utils.serializer._

import scala.reflect.ClassTag
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.runtime._

class Seq2seq[T: ClassTag](val encoderRecs: Array[Recurrent[T]],
   val decoderRecs: Array[Recurrent[T]],
   val preEncoder: AbstractModule[Activity, Activity, T] = null,
   val preDecoder: AbstractModule[Activity, Activity, T] = null,
   val shrinkHiddenStateModules: Array[Array[TensorModule[T]]] = null,
   val decoderInputType: DecoderInputType = DecoderInputType.ENCODERINPUTLASTTIME)
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
            hiddenState(i + 1) = shrinkHiddenState(rec.getHiddenState(),
              shrinkHiddenStateModules(i))
          } else hiddenState(i + 1) = rec.getHiddenState()
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
//      if (preDecoderInput.dim == decoderGradInput.dim) {
        preDecoder.backward(preDecoderInput, decoderGradInput)
//      } else {
//        preDecoder.backward(preDecoderInput, decoderGradInput.select(2, 1).contiguous())
//      }
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
                gradHiddenStates.toTable(i + 1), shrinkHiddenStateModules(i))
              x.setGradHiddenState(newGradHiddenState)
          } else x.setGradHiddenState(gradHiddenStates.toTable(i + 1))
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

  override def reset(): Unit = module.reset()

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

  override def clearState() : this.type = {
    super.clearState()
    preDecoderInput = null
    decoderInput = null
    encoderInput = null
    encoderOutput = null
    module.clearState()
    this
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

object Seq2seq extends ModuleSerializable {
  def apply[@specialized(Float, Double) T: ClassTag](encoderCells: Array[Recurrent[T]],
     decoderCells: Array[Recurrent[T]], preEncoder: AbstractModule[Activity, Activity, T] = null,
     preDecoder: AbstractModule[Activity, Activity, T] = null,
     shrinkHiddenStateModules: Array[Array[TensorModule[T]]] = null,
     decoderInputType: DecoderInputType = DecoderInputType.ENCODERINPUTLASTTIME)
    (implicit ev: TensorNumeric[T]): Seq2seq[T] = {
    new Seq2seq[T](encoderCells, decoderCells, preEncoder, preDecoder,
      shrinkHiddenStateModules, decoderInputType)
  }

  override def doLoadModule[T: ClassTag](context: DeserializeContext)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {
    val attrMap = context.bigdlModule.getAttrMap
    val encoderCellsAttr = attrMap.get("encoderCells")
    val encoderCells = DataConverter.
      getAttributeValue(context, encoderCellsAttr).
      asInstanceOf[Array[AbstractModule[_, _, T]]].map(_.asInstanceOf[Recurrent[T]])

    val decoderCellsAttr = attrMap.get("decoderCells")
    val decoderCells = DataConverter.
      getAttributeValue(context, decoderCellsAttr).
      asInstanceOf[Array[AbstractModule[_, _, T]]].map(_.asInstanceOf[Recurrent[T]])

    val preEncoderAttr = attrMap.get("preEncoder")
    val preEncoder = DataConverter.
      getAttributeValue(context, preEncoderAttr).
      asInstanceOf[AbstractModule[Activity, Activity, T]]

    val preDecoderAttr = attrMap.get("preDecoder")
    val preDecoder = DataConverter.
      getAttributeValue(context, preDecoderAttr).
      asInstanceOf[AbstractModule[Activity, Activity, T]]

    val shrinkHiddenStateAttr = attrMap.get("shrinkHiddenStateModules")
    val shrinkHiddenState = DataConverter.
      getAttributeValue(context, shrinkHiddenStateAttr).
      asInstanceOf[Array[AbstractModule[_, _, T]]]
    val shrinkHiddenStateModulesFlat = if (shrinkHiddenState == null) null
      else shrinkHiddenState.map(_.asInstanceOf[TensorModule[T]])
    var shrinkHiddenStateModules: Array[Array[TensorModule[T]]] = null
    if (shrinkHiddenStateModulesFlat != null) {
      val shrinkHiddenStateModulesBuffer = new ArrayBuffer[Array[TensorModule[T]]]()
      var i = 0
      while (i < shrinkHiddenStateModulesFlat.size) {
        shrinkHiddenStateModulesBuffer += shrinkHiddenStateModulesFlat.slice(i, i + 2)
        i += 2
      }
      shrinkHiddenStateModules = shrinkHiddenStateModulesBuffer.toArray
    }

    val decoderInputTypeAttr = attrMap.get("DecoderInputType")
    val decoderInputTypeStr = DataConverter.
      getAttributeValue(context, decoderInputTypeAttr).
      asInstanceOf[String]
    val decoderInputType = if (decoderInputTypeStr.equals("ENCODERINPUTSPLIT")) {
      DecoderInputType.ENCODERINPUTSPLIT
    } else {
      DecoderInputType.ENCODERINPUTLASTTIME
    }

    Seq2seq(encoderCells, decoderCells, preEncoder, preDecoder,
      shrinkHiddenStateModules, decoderInputType)
  }

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
    seq2seqBuilder : BigDLModule.Builder)
   (implicit ev: TensorNumeric[T]) : Unit = {
    val seq2seq = context.moduleData.module.asInstanceOf[Seq2seq[T]]

    val encoderRecsBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context,
      encoderRecsBuilder, seq2seq.encoderRecs,
      universe.typeOf[Array[_ <:
        AbstractModule[_ <: Activity, _ <:  Activity, _ <: Any]]]
    )
    seq2seqBuilder.putAttr("encoderCells", encoderRecsBuilder.build)

    val decoderRecsBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context,
    decoderRecsBuilder, seq2seq.decoderRecs,
      universe.typeOf[Array[_ <:
        AbstractModule[_ <: Activity, _ <:  Activity, _ <: Any]]])
    seq2seqBuilder.putAttr("decoderCells", decoderRecsBuilder.build)

      val preEncoderBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(context,
        preEncoderBuilder, seq2seq.preEncoder,
        ModuleSerializer.abstractModuleType)
      seq2seqBuilder.putAttr("preEncoder", preEncoderBuilder.build)

      val preDecoderBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(context,
        preDecoderBuilder, seq2seq.preDecoder,
        ModuleSerializer.abstractModuleType)
      seq2seqBuilder.putAttr("preDecoder", preDecoderBuilder.build)

    val module = if (seq2seq.shrinkHiddenStateModules == null) null
      else seq2seq.shrinkHiddenStateModules.flatten
      val shrinkHiddenStateModulesBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(context,
        shrinkHiddenStateModulesBuilder, module,
    universe.typeOf[Array[_ <:
      AbstractModule[_ <: Activity, _ <:  Activity, _ <: Any]]])
      seq2seqBuilder.putAttr("shrinkHiddenStateModules", shrinkHiddenStateModulesBuilder.build)

    val inputType = seq2seq.decoderInputType.toString
    val decoderInputTypeBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context,
      decoderInputTypeBuilder, inputType,
      universe.typeOf[String])
    seq2seqBuilder.putAttr("DecoderInputType", decoderInputTypeBuilder.build)
  }
}

object DecoderInputType extends Enumeration {
  type DecoderInputType = Value
  val ENCODERINPUTSPLIT, ENCODERINPUTLASTTIME = Value
}
