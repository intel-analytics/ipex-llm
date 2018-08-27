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

import com.intel.analytics.bigdl.mkl._
import com.intel.analytics.bigdl.nn.abstractnn.{Activity, Initializable}
import com.intel.analytics.bigdl.nn.{InitializationMethod, RandomUniform, VariableFormat}
import com.intel.analytics.bigdl.tensor._

import scala.collection.mutable.ArrayBuffer

class Linear(
  val inputSize: Int,
  val outputSize: Int,
  private val initWeight: Tensor[Float] = null,
  private val initBias: Tensor[Float] = null,
  private val initGradWeight: Tensor[Float] = null,
  private val initGradBias: Tensor[Float] = null) extends MklDnnLayer with Initializable {

  private[mkldnn] val weight: Blob = new Blob(Array(outputSize, inputSize))
  private[mkldnn] val bias: Blob = new Blob(Array(outputSize))
  private[mkldnn] val gradWeight: Blob = new Blob(Array(outputSize, inputSize))
  private[mkldnn] val gradBias: Blob = new Blob(Array(outputSize))

  @transient private var forwardPrimDesc: Long = 0L

  @transient private var updateOutputMemoryPrimitives: Array[Long] = _
  @transient private var updateOutputTensors: Array[Tensor[Float]] = _
  @transient private var updateGradInputMemoryPrimitives: Array[Long] = _
  @transient private var updateGradInputTensors: Array[Tensor[Float]] = _
  @transient private var updateGradWMemoryPrimitives: Array[Long] = _
  @transient private var updateGradWTensors: Array[Tensor[Float]] = _

  {
    val stdv = 1.0 / math.sqrt(weight.size(2))
    val wInit: InitializationMethod = RandomUniform(-stdv, stdv)
    val bInit: InitializationMethod = RandomUniform(-stdv, stdv)
    setInitMethod(wInit, bInit)
  }

  override def reset(): Unit = {
    if (initWeight == null) {
      weightInitMethod.init(weight.dense, VariableFormat.OUT_IN)
      weight.syncToNative()
    } else {
      weight.copy(initWeight)
    }

    if (initBias == null) {
      biasInitMethod.init(bias.dense, VariableFormat.ONE_D)
      bias.syncToNative()
    } else {
      bias.copy(initBias)
    }

    zeroGradParameters()
  }

  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    val weightShape = inputs(0).shape.length match {
      case 4 => Array(weight.size(1)) ++ inputs(0).shape.slice(1, 4)
      case _ => weight.size()
    }

    val inputShape = inputs(0).shape
    require(inputs(0).shape.length > 1, s"mkldnn linear unspported input dimension")

    val outputShape = Array(inputs(0).shape(0), outputSize)

    MklDnn.MemoryDescInit(inputShape.length, inputShape,
      DataType.F32, Memory.Format.any)

    val src = NativeData(inputShape, Memory.Format.any)
    val wei = NativeData(weightShape, Memory.Format.any)
    val bis = NativeData(bias.size(), Memory.Format.x)
    val dst = NativeData(outputShape, Memory.Format.any)

    val desc = MklDnn.LinearForwardDescInit(
      PropKind.Forward,
      src.getMemoryDescription(),
      wei.getMemoryDescription(),
      bis.getMemoryDescription(),
      dst.getMemoryDescription())
    forwardPrimDesc = MklDnn.PrimitiveDescCreate(desc, runtime.engine, 0)

    val List(realSrc, realWei, realDst) = List(Query.SrcPd, Query.WeightsPd, Query.DstPd).map {x =>
      MemoryData.operationWant(forwardPrimDesc, x)
    }

    require(weight.size().product == realWei.shape.product,
      s"${getName} weight shape is not correct.")

    weight.setMemoryData(realWei)
    bias.setMemoryData(bis)

    val srcs = Array(realSrc.getPrimitive(runtime), realWei.getPrimitive(runtime),
      bis.getPrimitive(runtime))
    val indexes = Array.fill(srcs.length)(0)
    val dsts = Array(realDst.getPrimitive(runtime))

    val primitive = MklDnn.PrimitiveCreate2(forwardPrimDesc, srcs, indexes, srcs.length,
      dsts, dsts.length)

    updateOutputMemoryPrimitives = srcs ++ dsts
    updateOutputPrimitives = Array(primitive)
    output = initTensor(dst)

    _inputFormats = Array(realSrc)
    _outputFormats = Array(realDst)
    (_inputFormats, _outputFormats)
  }

  override def updateOutput(input: Activity): Activity = {
    if (updateOutputTensors == null) {
      val buffer = new ArrayBuffer[Tensor[Float]]()
      buffer.append(input.asInstanceOf[Tensor[Float]])
      buffer.append(weight.native)
      buffer.append(bias.native)
      buffer.append(output.asInstanceOf[Tensor[Float]])
      updateOutputTensors = buffer.toArray
    }

    updateWithNewTensor(updateOutputTensors, 0, input)

    if (isTraining()) {
      weight.syncToNative()
      bias.syncToNative()
    }

    MklDnnOps.streamSubmit(runtime.stream, 1, updateOutputPrimitives, updateOutputPrimitives.length,
      updateOutputMemoryPrimitives, updateOutputTensors)

    output
  }

  override private[mkldnn] def initBwdPrimitives(grad: Array[MemoryData], phase: Phase) = {
    val weightShape = inputFormats()(0).shape.length match {
      case 4 => Array(weight.size(1)) ++ inputFormats()(0).shape.slice(1, 4)
      case _ => weight.size()
    }

    val inputShape = inputFormats()(0).shape

    val outputShape = Array(inputFormats()(0).shape(0), outputSize)

    val src = NativeData(inputShape, Memory.Format.any)
    val wei = NativeData(weightShape, Memory.Format.any)
    val bis = NativeData(bias.size(), Memory.Format.x)
    val dst = NativeData(outputShape, Memory.Format.any)

    val desc = MklDnn.LinearBackwardDataDescInit(
      src.getMemoryDescription(),
      wei.getMemoryDescription(),
      grad(0).getMemoryDescription())
    val backwardPrimDesc = MklDnn.PrimitiveDescCreate(desc, runtime.engine, forwardPrimDesc)

    val List(realDiffSrc, realWei, realDiffDst) =
      List(Query.DiffSrcPd, Query.WeightsPd, Query.DiffDstPd).map { x =>
        MemoryData.operationWant(backwardPrimDesc, x)
      }

    val srcs = Array(realDiffDst.getPrimitive(runtime), realWei.getPrimitive(runtime))
    val indexes = Array.fill(srcs.length)(0)
    val dsts = Array(realDiffSrc.getPrimitive(runtime))

    val primitive = MklDnn.PrimitiveCreate2(backwardPrimDesc, srcs, indexes, srcs.length,
      dsts, dsts.length)

    updateGradInputMemoryPrimitives = srcs ++ dsts
    updateGradInputPrimitives = Array(primitive)
    gradInput = initTensor(realDiffSrc)

    _gradInputFormats = Array(realDiffSrc)
    _gradOutputFormats = Array(realDiffDst)
    (_gradOutputFormats, _gradInputFormats)
  }

  override private[mkldnn] def initGradWPrimitives(grad: Array[MemoryData],
    phase: Phase): Array[MemoryData] = {
    val weightShape = inputFormats()(0).shape.length match {
      case 4 => Array(weight.size(1)) ++ inputFormats()(0).shape.slice(1, 4)
      case _ => weight.size()
    }

    val inputShape = inputFormats()(0).shape

    val outputShape = Array(inputFormats()(0).shape(0), outputSize)


    val src = NativeData(inputShape, Memory.Format.any)
    val wei = NativeData(weightShape, Memory.Format.any)
    val bis = NativeData(bias.size(), Memory.Format.x)
    val dst = NativeData(outputShape, Memory.Format.any)

    val desc = MklDnn.LinearBackwardWeightsDescInit(
      src.getMemoryDescription(), wei.getMemoryDescription(), bis.getMemoryDescription(),
      dst.getMemoryDescription())
    val gradWeightPrimDesc = MklDnn.PrimitiveDescCreate(desc, runtime.engine, forwardPrimDesc)

    val List(realWei, realDiffDst) = List(Query.DiffWeightsPd, Query.DiffDstPd).map { x =>
      MemoryData.operationWant(gradWeightPrimDesc, x)
    }

    gradWeight.setMemoryData(realWei)
    gradBias.setMemoryData(bis)

    val srcs = Array(inputFormats()(0).getPrimitive(runtime), realDiffDst.getPrimitive(runtime))
    val indexes = Array.fill(srcs.length)(0)
    val dsts = Array(realWei.getPrimitive(runtime), bis.getPrimitive(runtime))

    val primitive = MklDnn.PrimitiveCreate2(gradWeightPrimDesc, srcs, indexes, srcs.length,
      dsts, dsts.length)

    updateGradWMemoryPrimitives = srcs ++ dsts
    accGradientPrimitives = Array(primitive)

    _gradOutputFormatsForWeight = Array(realDiffDst)
    (_gradOutputFormatsForWeight)
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    if (updateGradInputTensors == null) {
      val buffer = new ArrayBuffer[Tensor[Float]]()
      buffer.append(gradOutput.asInstanceOf[Tensor[Float]])
      buffer.append(weight.native)
      buffer.append(gradInput.asInstanceOf[Tensor[Float]])
      updateGradInputTensors = buffer.toArray
    }

    updateWithNewTensor(updateGradInputTensors, 0, gradOutput)

    MklDnnOps.streamSubmit(runtime.stream, 1, updateGradInputPrimitives,
      updateGradInputPrimitives.length, updateGradInputMemoryPrimitives, updateGradInputTensors)

    gradInput
  }

  override def accGradParameters(input: Activity, gradOutput: Activity): Unit = {
    if (updateGradWTensors == null) {
      val buffer = new ArrayBuffer[Tensor[Float]]()
      buffer.append(input.asInstanceOf[Tensor[Float]])
      buffer.append(gradOutput.asInstanceOf[Tensor[Float]])
      buffer.append(gradWeight.native)
      buffer.append(gradBias.native)
      updateGradWTensors = buffer.toArray
    }

    updateWithNewTensor(updateGradInputTensors, 0, input)
    updateWithNewTensor(updateGradInputTensors, 1, gradOutput)

    MklDnnOps.streamSubmit(runtime.stream, 1, accGradientPrimitives,
      accGradientPrimitives.length, updateGradWMemoryPrimitives, updateGradWTensors)

    gradWeight.syncToHeap()
    gradBias.syncToHeap()
  }

  override def parameters(): (Array[Tensor[Float]], Array[Tensor[Float]]) = {
    (Array(weight.dense, bias.dense), Array(gradWeight.dense, gradBias.dense))
  }

  override def parametersWithShape(): (Array[MemoryData], Array[MemoryData]) = {
    (Array(weight.memoryData(), bias.memoryData()),
      Array(gradWeight.memoryData(), gradBias.memoryData()))
  }

  override def zeroGradParameters(): Unit = {
    gradWeight.zero()
    gradBias.zero()
  }

  override def release(): Unit = {
    super.release()
    List(weight, bias, gradWeight, gradBias).foreach(_.release())
  }
}

object Linear {
  def apply(
    inputSize: Int,
    outputSize: Int,
    withBias: Boolean = true,
    initWeight: Tensor[Float] = null,
    initBias: Tensor[Float] = null,
    initGradWeight: Tensor[Float] = null,
    initGradBias: Tensor[Float] = null): Linear = {
    new Linear(inputSize, outputSize, initWeight, initBias, initGradWeight, initGradBias)
  }
}
