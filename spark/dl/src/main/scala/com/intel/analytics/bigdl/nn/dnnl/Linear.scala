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

import com.intel.analytics.bigdl.dnnl._
import com.intel.analytics.bigdl.nn.abstractnn.{Activity, Initializable}
import com.intel.analytics.bigdl.nn.{InitializationMethod, MklInt8Convertible, RandomUniform, VariableFormat}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor._

import scala.collection.mutable

class Linear(
  val inputSize: Int,
  val outputSize: Int,
  var wRegularizer: Regularizer[Float] = null,
  var bRegularizer: Regularizer[Float] = null,
  private val initWeight: Tensor[Float] = null,
  private val initBias: Tensor[Float] = null,
  private val initGradWeight: Tensor[Float] = null,
  private val initGradBias: Tensor[Float] = null)
  extends MklDnnLayer with Initializable with MklInt8Convertible {

  private[mkldnn] val weight: TensorMMap = new TensorMMap(Array(outputSize, inputSize))
  private[mkldnn] val bias: TensorMMap = new TensorMMap(Array(outputSize))
  private[mkldnn] val gradWeight: TensorMMap = new TensorMMap(Array(outputSize, inputSize))
  private[mkldnn] val gradBias: TensorMMap = new TensorMMap(Array(outputSize))

  @transient private var forwardPrimDesc: Long = 0L
  @transient private var updateGradWMemoryPrimitives: Array[Long] = _
//  @transient private var updateGradWTensors: Array[Tensor[Float]] = _
  @transient private var updateGradWTensors: mutable.Map[Int, Tensor[Float]] = _

  {
    val stdv = 1.0 / math.sqrt(weight.size(2))
    val wInit: InitializationMethod = RandomUniform(-stdv, stdv)
    val bInit: InitializationMethod = RandomUniform(-stdv, stdv)
    setInitMethod(wInit, bInit)
  }

  override def reset(): Unit = {
    if (initWeight == null) {
      weightInitMethod.init(weight.dense, VariableFormat.OUT_IN)
    } else {
      weight.dense.copy(initWeight)
    }

    if (initBias == null) {
      biasInitMethod.init(bias.dense, VariableFormat.ONE_D)
    } else {
      bias.dense.copy(initBias)
    }
  }


  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    require(inputs(0).shape.length > 1, s"mkldnn linear unspported input dimension")

    val (weightShape, weightLayout) = inputs(0).shape.length match {
      case 4 =>
        (Array(weight.size(1)) ++ inputs(0).shape.slice(1, 4),
          Memory.FormatTag.oihw)
      case 2 => (weight.size(), Memory.FormatTag.nc)
      case 1 => (weight.size(), Memory.FormatTag.x)
    }

    val inputShape = inputs(0).shape
    val outputShape = Array(inputs(0).shape(0), outputSize)

    val src = NativeData(inputShape, Memory.FormatTag.any)
    val wei = NativeData(weightShape, Memory.FormatTag.any)
    val bis = NativeData(bias.size(), Memory.FormatTag.x)
    val dst = NativeData(outputShape, Memory.FormatTag.any)

    val desc = DnnlMemory.LinearForwardDescInit(
      PropKind.Forward,
      src.getMemoryDescriptor(),
      wei.getMemoryDescriptor(),
      bis.getMemoryDescriptor(),
      dst.getMemoryDescriptor())

    forwardPrimDesc = DnnlMemory.PrimitiveDescCreate(desc, runtime.engine, 0)

    val List(realSrc, realWei, realDst) = List(Query.SrcMd, Query.WeightsMd, Query.DstMd).map {x =>
      MemoryData.operationWant(forwardPrimDesc, x)
    }

    require(weight.size().product == realWei.shape.product,
      s"${getName} weight shape is not correct." + weight.size().mkString(" ") + "   " +
    realWei.shape.mkString(" "))

    weight.setMemoryData(HeapData(weightShape, weightLayout), realWei, runtime)
    bias.setMemoryData(HeapData(bis.shape, Memory.FormatTag.x), bis, runtime)
    weight.sync()
    bias.sync()

    val primitive = DnnlMemory.PrimitiveCreate(forwardPrimDesc)

    updateOutputPrimitives = Array(primitive)
    output = initTensor(realDst)
    _inputFormats = Array(realSrc)
    _outputFormats = Array(realDst)

    fwdExecArgs = mutable.Map(
      ArgType.DNNL_ARG_SRC -> realSrc.getMemoryObject(runtime),
      ArgType.DNNL_ARG_WEIGHTS -> realWei.getMemoryObject(runtime),
      ArgType.DNNL_ARG_BIAS -> bis.getMemoryObject(runtime),
      ArgType.DNNL_ARG_DST -> realDst.getMemoryObject(runtime)
    )

    (_inputFormats, _outputFormats)
  }

  override def updateOutput(input: Activity): Activity = {
    if (updateOutputTensors == null) {
      updateOutputTensors = mutable.Map(
        ArgType.DNNL_ARG_SRC -> input.asInstanceOf[Tensor[Float]],
        ArgType.DNNL_ARG_WEIGHTS -> weight.native,
        ArgType.DNNL_ARG_BIAS -> bias.native,
        ArgType.DNNL_ARG_DST -> output.asInstanceOf[Tensor[Float]]
      )
    }
    updateWithNewTensor(updateOutputTensors, ArgType.DNNL_ARG_SRC, input)

    if (isTraining()) {
      weight.sync()
      bias.sync()
    }

    MklDnnOps.streamSubmit(updateOutputPrimitives,
      runtime.stream,
      fwdExecArgs,
      updateOutputTensors)

    output
  }

  override private[mkldnn] def initBwdPrimitives(grad: Array[MemoryData], phase: Phase) = {
    val weightShape = inputFormats()(0).shape.length match {
      case 4 => Array(weight.size(1)) ++ inputFormats()(0).shape.slice(1, 4)
      case _ => weight.size()
    }

    val inputShape = inputFormats()(0).shape
    val src = NativeData(inputShape, Memory.FormatTag.any)
    val wei = NativeData(weightShape, Memory.FormatTag.any)

    val desc = DnnlMemory.LinearBackwardDataDescInit(
      src.getMemoryDescriptor(),
      wei.getMemoryDescriptor(),
      grad(0).getMemoryDescriptor())

    val backwardPrimDesc = DnnlMemory.PrimitiveDescCreate(desc, runtime.engine, forwardPrimDesc)

    val List(realDiffSrc, realWei, realDiffDst) =
      List(Query.DiffSrcMd, Query.WeightsMd, Query.DiffDstMd).map { x =>
        MemoryData.operationWant(backwardPrimDesc, x)
      }

    val primitive = DnnlMemory.PrimitiveCreate(backwardPrimDesc)

    updateGradInputPrimitives = Array(primitive)

    gradInput = initTensor(realDiffSrc)

    _gradInputFormats = Array(realDiffSrc)
    _gradOutputFormats = Array(realDiffDst)

    bwdExecArgs = mutable.Map(
      ArgType.DNNL_ARG_DIFF_SRC -> realDiffSrc.getMemoryObject(runtime),
      ArgType.DNNL_ARG_DIFF_DST -> realDiffDst.getMemoryObject(runtime),
      ArgType.DNNL_ARG_WEIGHTS -> realWei.getMemoryObject(runtime)
    )

    (_gradOutputFormats, _gradInputFormats)
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    if (updateGradInputTensors == null) {
      updateGradInputTensors = mutable.Map(
        ArgType.DNNL_ARG_DIFF_SRC -> gradInput.asInstanceOf[Tensor[Float]],
        ArgType.DNNL_ARG_DIFF_DST -> gradOutput.asInstanceOf[Tensor[Float]],
        ArgType.DNNL_ARG_WEIGHTS -> weight.dense
      )
    }

    updateWithNewTensor(updateGradInputTensors, ArgType.DNNL_ARG_DIFF_DST, gradOutput)

    MklDnnOps.streamSubmit(updateGradInputPrimitives,
      runtime.stream, bwdExecArgs, updateGradInputTensors)

    gradInput
  }

  override private[mkldnn] def initGradWPrimitives(grad: Array[MemoryData],
                                                 phase: Phase): Array[MemoryData] = {
    val (weightShape, weightLayout) = inputFormats()(0).shape.length match {
      case 4 =>
        (Array(weight.size(1)) ++ inputFormats()(0).shape.slice(1, 4),
          Memory.FormatTag.oihw)
      case 2 => (weight.size(), Memory.FormatTag.nc)
      case 1 => (weight.size(), Memory.FormatTag.x)
    }

    val inputShape = inputFormats()(0).shape
    val outputShape = Array(inputFormats()(0).shape(0), outputSize)

    val src = NativeData(inputShape, Memory.FormatTag.any)
    val wei = NativeData(weightShape, Memory.FormatTag.any)
    val bis = NativeData(bias.size(), Memory.FormatTag.x)
    val dst = NativeData(outputShape, Memory.FormatTag.any)

    val desc = DnnlMemory.LinearBackwardWeightsDescInit(
      src.getMemoryDescriptor(),
      wei.getMemoryDescriptor(),
      bis.getMemoryDescriptor(),
      dst.getMemoryDescriptor()
    )

    val gradWeightPrimDesc = DnnlMemory.PrimitiveDescCreate(desc, runtime.engine, forwardPrimDesc)

    val List(realWei, realDiffDst) = List(Query.DiffWeightsMd, Query.DiffDstMd).map { x =>
      MemoryData.operationWant(gradWeightPrimDesc, x)
    }

    gradWeight.setMemoryData(realWei, HeapData(weightShape, weightLayout), runtime)
    gradBias.setMemoryData(bis, HeapData(bis.shape, Memory.FormatTag.x), runtime)
    gradWeight.zero()
    gradBias.zero()

    val primitive = DnnlMemory.PrimitiveCreate(gradWeightPrimDesc)

    weightUpdateExecArgs = mutable.Map(
      ArgType.DNNL_ARG_SRC -> inputFormats()(0).getMemoryObject(runtime),
      ArgType.DNNL_ARG_DIFF_DST -> realDiffDst.getMemoryObject(runtime),
      ArgType.DNNL_ARG_DIFF_WEIGHTS -> realWei.getMemoryObject(runtime),
      ArgType.DNNL_ARG_DIFF_BIAS -> bis.getMemoryObject(runtime)
    )

    accGradientPrimitives = Array(primitive)

    _gradOutputFormatsForWeight = Array(realDiffDst)

    (_gradOutputFormatsForWeight)
  }

  override def accGradParameters(input: Activity, gradOutput: Activity): Unit = {
    if (updateGradWTensors == null) {
      updateGradWTensors = mutable.Map(
        ArgType.DNNL_ARG_SRC -> input.asInstanceOf[Tensor[Float]],
        ArgType.DNNL_ARG_DIFF_DST -> gradOutput.asInstanceOf[Tensor[Float]],
        ArgType.DNNL_ARG_DIFF_WEIGHTS -> gradWeight.native,
        ArgType.DNNL_ARG_DIFF_BIAS -> gradBias.native
      )
    }

    updateWithNewTensor(updateGradWTensors, ArgType.DNNL_ARG_SRC, input)
    updateWithNewTensor(updateGradWTensors, ArgType.DNNL_ARG_DIFF_DST, gradOutput)

    MklDnnOps.streamSubmit(accGradientPrimitives,
      runtime.stream, weightUpdateExecArgs, updateGradWTensors)

    gradWeight.sync()
    gradBias.sync()

    if (null != wRegularizer && scaleW != 0) {
      wRegularizer.accRegularization(weight.dense, gradWeight.dense, scaleW)
    }
    if (null != bRegularizer && scaleB != 0) {
      bRegularizer.accRegularization(bias.dense, gradBias.dense, scaleB)
    }
  }

  override def parameters(): (Array[Tensor[Float]], Array[Tensor[Float]]) = {
    (Array(weight.dense, bias.dense), Array(gradWeight.dense, gradBias.dense))
  }

  override def paramsMMap(): (Array[TensorMMap], Array[TensorMMap]) = {
    (Array(weight, bias), Array(gradWeight, gradBias))
  }

  override def zeroGradParameters(): Unit = {
  }

}

object Linear {
  def apply(
    inputSize: Int,
    outputSize: Int,
    withBias: Boolean = true,
    wRegularizer: Regularizer[Float] = null,
    bRegularizer: Regularizer[Float] = null,
    initWeight: Tensor[Float] = null,
    initBias: Tensor[Float] = null,
    initGradWeight: Tensor[Float] = null,
    initGradBias: Tensor[Float] = null): Linear = {
    new Linear(inputSize, outputSize, wRegularizer,
      bRegularizer, initWeight, initBias, initGradWeight, initGradBias)
  }
}
