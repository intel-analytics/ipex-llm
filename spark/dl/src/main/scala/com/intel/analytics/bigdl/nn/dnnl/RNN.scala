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
import com.intel.analytics.bigdl.nn.{InitializationMethod, RandomUniform, VariableFormat, Zeros}
import com.intel.analytics.bigdl.nn.abstractnn.{Activity, Initializable}
import com.intel.analytics.bigdl.tensor.{DnnTensor, Tensor}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
 * @param mode       : the type of RNN cell (LSTM / GRU)
 * @param inputSize  : the size of input vector
 * @param hiddenSize : the size of hidden state
 * @param f          : the type of output activation function
 *                   (AlgKind.EltwiseTanh or AlgKind.EltwiseRelu)
 * @param direction  : the direction to run RNN
 *                   (e.g. Direction.UnidirectionalLeft2Right or Direction.BidirectionalConcat)
 * @param layers     : the number of RNN layers
 */

class RNN(
  val mode: Int,
  val inputSize: Int,
  val hiddenSize: Int,
  val f: Int,
  val direction: Int,
  val layers: Int = 1,
  val flags: Int = 0, // TODO: RNNCellFlags.RNNCellWithRelu,
  val alpha: Float = 0F,
  val clipping: Float = 0F,
  private val initWeight: Tensor[Float] = null,
  private val initWeightIter: Tensor[Float] = null,
  private val initBias: Tensor[Float] = null
) extends MklDnnLayer with Initializable {
//  private var updateOutputMemoryPrimitives: Array[Long] = _
//  private var updateOutputTensors: Array[Tensor[Float]] = _
//  private var updateGradInputMemoryPrimitives: Array[Long] = _
//  private var updateGradInputTensors: Array[Tensor[Float]] = _
  private var fwdPD: Long = _
  private var rnnCellDesc : Long = 0L

  private[mkldnn] var weight: TensorMMap = _
  private[mkldnn] var weight_i: TensorMMap = _
  private[mkldnn] var bias: TensorMMap = _
  private[mkldnn] var gradWeight: TensorMMap = _
  private[mkldnn] var gradWeight_i: TensorMMap = _
  private[mkldnn] var gradBias: TensorMMap = _

  private var workSpaceFormat: MemoryData = _
  private var workSpace : Tensor[Float] = _

  @transient private lazy val reorderManager = new ReorderManager
  private var weightForBackward: DnnTensor[Float] = _
  private var weightForBackwardMemoryData: MemoryData = _
  private var weightIterForBackward: DnnTensor[Float] = _
  private var weightIterForBackwardMemoryData: MemoryData = _

  private var batchSize: Int = _
  private var stepSize: Int = _
  private var inputShape: Array[Int] = _
  private var outputShape: Array[Int] = _
  private var weightShape: Array[Int] = _
  private var weightIterShape: Array[Int] = _
  private var biasShape: Array[Int] = _
  private var commonIterShape: Array[Int] = _

  private var src_i: DnnTensor[Float] = _
  private var dst_i: DnnTensor[Float] = _
  private var gradsrc_i: DnnTensor[Float] = _
  private var graddst_i: DnnTensor[Float] = _

  if(layers > 1) {
    require(inputSize == hiddenSize,
      "If layer number of RNN is more than 1, the input size and the hidden size should equal.\n"
      + "inputSize: " + inputSize + '\n'
      + "hiddenSize: " + hiddenSize)
  }

  var (ngates, nstates) = mode match {
    case AlgKind.VanillaLstm => (4, 2)
    case AlgKind.VanillaGru => (3, 1)
    case _ =>
      throw new UnsupportedOperationException("Not support such RNN Cell. Cell type: " + mode)
  }

  /** TODO: Multi-layer Bidirectional Sum RNN is available in MKLDNN,
   *  TODO: but the current version of BigDL BLAS does not support it.
   */

  val (numOfDirections, outputSizeFactor) = direction match {
    case Direction.UnidirectionalLeft2Right
         | Direction.UnidirectionalRight2Left => (1, 1)
    case Direction.BidirectionalConcat =>
      require(layers == 1, "Bidirectional Concat RNN does not support multiple layers. " +
        "layers = " + layers)
      (2, 2)
    case Direction.BidirectionalSum => (2, 1)
    case _ => throw new UnsupportedOperationException("Not support such direction")
  }

  /**
   * Gate order matching between MKLDNN LSTM and nn/LSTM:
   * MKLDNN Gate 1 -> nn/LSTM Gate 1 (input gate)
   * MKLDNN Gate 2 -> nn/LSTM Gate 3 (forget gate)
   * MKLDNN Gate 3 -> nn/LSTM Gate 2 (hidden)
   * MKLDNN Gate 4 -> nn/LSTM Gate 4 (output gate)
   *
   * Gate order matching between MKLDNN GRU and nn/GRU:
   * MKLDNN Gate 1 -> nn/GRU Gate 2
   * MKLDNN Gate 2 -> nn/GRU Gate 1
   * MKLDNN Gate 3 -> nn/GRU Gate 3
   */

  weightShape = Array(layers, numOfDirections, inputSize, ngates, hiddenSize)
  weightIterShape = Array(layers, numOfDirections, hiddenSize, ngates, hiddenSize)
  biasShape = Array(layers, numOfDirections, ngates, hiddenSize)

  weight = new TensorMMap(weightShape)
  weight_i = new TensorMMap(weightIterShape)
  bias = new TensorMMap(biasShape)
  gradWeight = new TensorMMap(weightShape)
  gradWeight_i = new TensorMMap(weightIterShape)
  gradBias = new TensorMMap(biasShape)

  {
    val stdv = 1.0 / math.sqrt(hiddenSize)
    val wInit: InitializationMethod = RandomUniform(-stdv, stdv)
    val bInit: InitializationMethod = Zeros
    setInitMethod(wInit, bInit)
  }

  override def reset(): Unit = {
    if (initWeight == null) {
      weightInitMethod.init(weight.dense, VariableFormat.Default)
    } else {
      weight.dense.copy(initWeight)
    }

    if (initWeightIter == null) {
      weightInitMethod.init(weight_i.dense, VariableFormat.Default)
    } else {
      weight_i.dense.copy(initWeightIter)
    }

    if (initBias == null) {
      biasInitMethod.init(bias.dense, VariableFormat.Default)
    } else {
      bias.dense.copy(initBias)
    }
  }

  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    val kind = if (!isTraining()) {
      PropKind.ForwardInference
    } else {
      PropKind.ForwardTraining
    }

    /**
     * TODO: The default format of input is TNC
     * Batch size of input is needed by creating memory descriptors of src iter and dst iter.
     * Step size of input is needed by creating memory descriptor of dst layer.
     * By default, batch size of input is the second element of inputShape
     * and step size is the first element of inputShape.
     */

    inputs(0).layout match {
      case Memory.FormatTag.tnc =>
        batchSize = inputs(0).shape(1)
        stepSize = inputs(0).shape(0)
      case Memory.FormatTag.ntc =>
        batchSize = inputs(0).shape(0)
        stepSize = inputs(0).shape(1)
      case _ =>
        throw new UnsupportedOperationException("Unsupported input format: " +
          inputs(0).layout)
    }

    inputShape = Array(stepSize, batchSize, inputSize)
    outputShape = Array(stepSize, batchSize, outputSizeFactor * hiddenSize)
    commonIterShape = Array(layers, numOfDirections, nstates, batchSize, hiddenSize)

    val src_layer = NativeData(inputShape, Memory.FormatTag.any)
    val src_iter = NativeData(commonIterShape, Memory.FormatTag.any)
    val wei_layer = NativeData(weightShape, Memory.FormatTag.any)
    val wei_iter = NativeData(weightIterShape, Memory.FormatTag.any)
    val bis = NativeData(biasShape, Memory.FormatTag.any)
    val dst_layer = NativeData(outputShape, Memory.FormatTag.any)
    val dst_iter = NativeData(commonIterShape, Memory.FormatTag.any)

    val src_layer_MD = src_layer.getMemoryDescriptor()
    val src_iter_MD = src_iter.getMemoryDescriptor()
    val weights_layer_MD = wei_layer.getMemoryDescriptor()
    val weights_iter_MD = wei_iter.getMemoryDescriptor()
    val bis_MD = bis.getMemoryDescriptor()
    val dist_layer_MD = dst_layer.getMemoryDescriptor()
    val dist_iter_MD = dst_iter.getMemoryDescriptor()

    rnnCellDesc = mode match {
      case AlgKind.VanillaLstm =>
        DnnlMemory.RNNCellDescInit(AlgKind.VanillaLstm, f, flags, alpha, clipping)
      case AlgKind.VanillaGru =>
        DnnlMemory.RNNCellDescInit(AlgKind.VanillaGru, f, flags, alpha, clipping)
      case _ => throw new UnsupportedOperationException("Not support such RNN cell. " +
        "Cell type: " + mode)
    }

    val description = DnnlMemory.RNNForwardDescInit(kind, rnnCellDesc, direction, src_layer_MD,
      src_iter_MD, weights_layer_MD, weights_iter_MD, bis_MD, dist_layer_MD, dist_iter_MD)

    fwdPD = DnnlMemory.PrimitiveDescCreate(description, runtime.engine, 0L)

    val realSrc = MemoryData.operationWant(fwdPD, Query.SrcMd, 0)
    val realSrc_iter = MemoryData.operationWant(fwdPD, Query.SrcMd, 1)
    val realWei = MemoryData.operationWant(fwdPD, Query.WeightsMd, 0)
    val realWei_iter = MemoryData.operationWant(fwdPD, Query.WeightsMd, 1)
    val realBias = MemoryData.operationWant(fwdPD, Query.WeightsMd, 2)
    val realDst = MemoryData.operationWant(fwdPD, Query.DstMd, 0)
    val realDst_iter = MemoryData.operationWant(fwdPD, Query.DstMd, 1)

    require(weight.size().product == realWei.shape.product,
      s"${getName} weight shape is not correct.")
    require(weight_i.size().product == realWei_iter.shape.product,
      s"${getName} weight iter shape is not correct.")
    require(bias.size().product == realBias.shape.product,
      s"${getName} bias shape is not correct.")

    weight.setMemoryData(HeapData(weightShape, Memory.FormatTag.ldigo), realWei, runtime)
    weight_i.setMemoryData(HeapData(weightIterShape, Memory.FormatTag.ldigo), realWei_iter, runtime)
    bias.setMemoryData(HeapData(biasShape, Memory.FormatTag.ldgo), realBias, runtime)

    weight.sync()
    weight_i.sync()
    bias.sync()

    src_i = initTensor(realSrc_iter).asInstanceOf[DnnTensor[Float]]
    dst_i = initTensor(realDst_iter).asInstanceOf[DnnTensor[Float]]
    src_i.zero()
    dst_i.zero()

//    val srcs = Array(
//      realSrc.getMemoryObject(runtime), realSrc_iter.getMemoryObject(runtime),
//      realWei.getMemoryObject(runtime), realWei_iter.getMemoryObject(runtime),
//      realBias.getMemoryObject(runtime))
//
//    val indexes = Array.fill(srcs.length)(0)
//
//    val dsts = if (isTraining()) {
//      Array(realDst.getMemoryObject(runtime),
//        realDst_iter.getMemoryObject(runtime),
//        workSpaceFormat.getMemoryObject(runtime))
//    }
//    else {
//      Array(realDst.getMemoryObject(runtime),
//        realDst_iter.getMemoryObject(runtime))
//    }

    fwdExecArgs = mutable.Map(
      ArgType.DNNL_ARG_SRC -> realSrc.getMemoryObject(runtime),
      ArgType.DNNL_ARG_SRC_ITER -> realSrc_iter.getMemoryObject(runtime),
      ArgType.DNNL_ARG_WEIGHTS -> realWei.getMemoryObject(runtime),
      ArgType.DNNL_ARG_WEIGHTS_ITER -> realWei_iter.getMemoryObject(runtime),
      ArgType.DNNL_ARG_BIAS -> realBias.getMemoryObject(runtime),
      ArgType.DNNL_ARG_DST -> realDst.getMemoryObject(runtime),
      ArgType.DNNL_ARG_DST_ITER -> realDst_iter.getMemoryObject(runtime)
    )

    if (isTraining()) {
      workSpaceFormat = MemoryData.operationWant(fwdPD, Query.WorkspaceMd, 0)
      workSpace = initTensor(workSpaceFormat).asInstanceOf[Tensor[Float]]
      fwdExecArgs(ArgType.DNNL_ARG_WORKSPACE) = workSpaceFormat.getMemoryObject(runtime)
    }

//    val primitive = DnnlMemory.PrimitiveCreate2(fwdPD, srcs, indexes,
//      srcs.length, dsts, dsts.length)
    val primitive = DnnlMemory.PrimitiveCreate(fwdPD)

    updateOutputPrimitives = Array(primitive)
    output = initTensor(realDst)

    _inputFormats = Array(realSrc)
    _outputFormats = Array(realDst)

    (_inputFormats, _outputFormats)
  }

  override def updateOutput(input: Activity): Activity = {
    if (updateOutputTensors == null) {
      updateOutputTensors = mutable.Map(
        ArgType.DNNL_ARG_SRC -> input.asInstanceOf[Tensor[Float]],
        ArgType.DNNL_ARG_SRC_ITER -> src_i,
        ArgType.DNNL_ARG_WEIGHTS -> weight.native,
        ArgType.DNNL_ARG_WEIGHTS_ITER -> weight_i.native,
        ArgType.DNNL_ARG_BIAS -> bias.native,
        ArgType.DNNL_ARG_DST -> output.asInstanceOf[Tensor[Float]],
        ArgType.DNNL_ARG_DST_ITER -> dst_i
      )
      if (isTraining()) {
        updateOutputTensors(ArgType.DNNL_ARG_WORKSPACE) = workSpace
      }
    }

    if (isTraining()) {
      weight.sync()
      weight_i.sync()
      bias.sync()
    }

    updateWithNewTensor(updateOutputTensors, ArgType.DNNL_ARG_SRC, input)

    // TODO:
    MklDnnOps.streamSubmit(updateOutputPrimitives,
      runtime.stream, fwdExecArgs, updateOutputTensors)

    output
  }

  override private[mkldnn] def initBwdPrimitives(grad: Array[MemoryData], phase: Phase) = {
    reorderManager.setRuntime(runtime)

    val src_layer_bw = NativeData(inputShape, Memory.FormatTag.any)
    val src_iter_bw = NativeData(commonIterShape, Memory.FormatTag.any)
    val wei_layer_bw = NativeData(weightShape, Memory.FormatTag.any)
    val wei_iter_bw = NativeData(weightIterShape, Memory.FormatTag.any)
    val bis_bw = NativeData(biasShape, Memory.FormatTag.any)
    val dst_layer_bw = NativeData(outputShape, Memory.FormatTag.any)
    val dst_iter_bw = NativeData(commonIterShape, Memory.FormatTag.any)
    val diff_src_layer = NativeData(inputShape, Memory.FormatTag.any)
    val diff_src_iter = NativeData(commonIterShape, Memory.FormatTag.any)
    val diff_weights_layer = NativeData(weightShape, Memory.FormatTag.ldigo)
    // IMPORTANT : it has to be ldigo
    val diff_weights_iter = NativeData(weightIterShape, Memory.FormatTag.ldigo)
    // IMPORTANT : it has to be ldigo
    val diff_bias = NativeData(biasShape, Memory.FormatTag.any)
    val diff_dist_layer = NativeData(outputShape, Memory.FormatTag.any)
    val diff_dist_iter = NativeData(commonIterShape, Memory.FormatTag.any)

    val src_layer_bw_MD = src_layer_bw.getMemoryDescriptor()
    val src_iter_bw_MD = src_iter_bw.getMemoryDescriptor()
    val weights_layer_bw_MD = wei_layer_bw.getMemoryDescriptor()
    val weights_iter_bw_MD = wei_iter_bw.getMemoryDescriptor()
    val bis_bw_MD = bis_bw.getMemoryDescriptor()
    val dist_layer_bw_MD = dst_layer_bw.getMemoryDescriptor()
    val dist_iter_bw_MD = dst_iter_bw.getMemoryDescriptor()
    val diff_src_layer_MD = diff_src_layer.getMemoryDescriptor()
    val diff_src_iter_MD = diff_src_iter.getMemoryDescriptor()
    val diff_weights_layer_MD = diff_weights_layer.getMemoryDescriptor()
    val diff_weights_iter_MD = diff_weights_iter.getMemoryDescriptor()
    val diff_bis_MD = diff_bias.getMemoryDescriptor()
    val diff_dist_layer_MD = diff_dist_layer.getMemoryDescriptor()
    val diff_dist_iter_MD = diff_dist_iter.getMemoryDescriptor()

    val description = DnnlMemory.RNNBackwardDescInit(PropKind.Backward, rnnCellDesc,
      direction, src_layer_bw_MD,
      src_iter_bw_MD, weights_layer_bw_MD,
      weights_iter_bw_MD, bis_bw_MD,
      dist_layer_bw_MD, dist_iter_bw_MD,

      diff_src_layer_MD, diff_src_iter_MD,
      diff_weights_layer_MD, diff_weights_iter_MD,
      diff_bis_MD, diff_dist_layer_MD,
      diff_dist_iter_MD
    )

    val bwdPD = DnnlMemory.PrimitiveDescCreate(description, runtime.engine, fwdPD)

    val realSrc = MemoryData.operationWant(bwdPD, Query.SrcMd, 0)
    val realSrc_iter = MemoryData.operationWant(bwdPD, Query.SrcMd, 1)
    val realWei = MemoryData.operationWant(bwdPD, Query.WeightsMd, 0)
    val realWei_iter = MemoryData.operationWant(bwdPD, Query.WeightsMd, 1)
    val realBias = MemoryData.operationWant(bwdPD, Query.WeightsMd, 2)
    val realDst = MemoryData.operationWant(bwdPD, Query.DstMd, 0)
    val realDst_iter = MemoryData.operationWant(bwdPD, Query.DstMd, 1)
    val realDiffDst = MemoryData.operationWant(bwdPD, Query.DiffDstMd, 0)
    val realDiffDst_iter = MemoryData.operationWant(bwdPD, Query.DiffDstMd, 1)

    val realDiffSrc = MemoryData.operationWant(bwdPD, Query.DiffSrcMd, 0)
    val realDiffSrc_iter = MemoryData.operationWant(bwdPD, Query.DiffSrcMd, 1)
    val realDiffWei = MemoryData.operationWant(bwdPD, Query.DiffWeightsMd, 0)
    val realDiffWei_iter = MemoryData.operationWant(bwdPD, Query.DiffWeightsMd, 1)
    val realDiffBias = MemoryData.operationWant(bwdPD, Query.DiffWeightsMd, 2)

    weightForBackwardMemoryData = realWei
    reorderManager.register(weight.heapData, realWei)
    weightForBackward = reorderManager
      .infer(Array(weight.heapData), Array(weightForBackwardMemoryData), weight.dense)
      .asInstanceOf[DnnTensor[Float]]

    weightIterForBackwardMemoryData = realWei_iter
    reorderManager.register(weight_i.heapData, realWei_iter)
    weightIterForBackward = reorderManager
      .infer(Array(weight_i.heapData), Array(weightIterForBackwardMemoryData), weight_i.dense)
      .asInstanceOf[DnnTensor[Float]]

    gradWeight.setMemoryData(realDiffWei, HeapData(weightShape, Memory.FormatTag.ldigo), runtime)
    gradWeight_i.setMemoryData(realDiffWei_iter, HeapData(weightIterShape, Memory.FormatTag.ldigo),
      runtime)
    gradBias.setMemoryData(realDiffBias, HeapData(biasShape, Memory.FormatTag.ldgo), runtime)

    gradWeight.zero()
    gradWeight_i.zero()
    gradBias.zero()

    gradsrc_i = initTensor(realDiffSrc_iter).asInstanceOf[DnnTensor[Float]]
    graddst_i = initTensor(realDiffDst_iter).asInstanceOf[DnnTensor[Float]]
    gradsrc_i.zero()
    graddst_i.zero()

    val srcs = Array(realSrc.getMemoryObject(runtime), realSrc_iter.getMemoryObject(runtime),
      realWei.getMemoryObject(runtime), realWei_iter.getMemoryObject(runtime),
      realBias.getMemoryObject(runtime), realDst.getMemoryObject(runtime),
      realDst_iter.getMemoryObject(runtime), realDiffDst.getMemoryObject(runtime),
      realDiffDst_iter.getMemoryObject(runtime), workSpaceFormat.getMemoryObject(runtime))
    val indexes = Array.fill(srcs.length)(0)

    val dsts = Array(realDiffSrc.getMemoryObject(runtime),
      realDiffSrc_iter.getMemoryObject(runtime),
      realDiffWei.getMemoryObject(runtime), realDiffWei_iter.getMemoryObject(runtime),
      realDiffBias.getMemoryObject(runtime)
    )

//    val primitive = DnnlMemory.PrimitiveCreate(bwdPD, srcs, indexes, srcs.length,
//      dsts, dsts.length)
    val primitive = DnnlMemory.PrimitiveCreate(bwdPD)

//    updateGradInputMemoryPrimitives = srcs ++ dsts
    updateGradInputPrimitives = Array(primitive)
    gradInput = initTensor(realDiffSrc)

    _gradInputFormats = Array(realDiffSrc)
    _gradOutputFormats = Array(realDiffDst)
    (_gradOutputFormats, _gradInputFormats)
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
//    if (updateGradInputTensors == null) {
//      val buffer = new ArrayBuffer[Tensor[Float]]()
//      buffer.append(input.asInstanceOf[Tensor[Float]])
//      buffer.append(src_i)
//      buffer.append(weightForBackward)
//      buffer.append(weightIterForBackward)
//      buffer.append(bias.native)
//      buffer.append(output.asInstanceOf[Tensor[Float]])
//      buffer.append(dst_i)
//      buffer.append(gradOutput.asInstanceOf[Tensor[Float]])
//      buffer.append(graddst_i)
//      buffer.append(workSpace)
//
//      buffer.append(gradInput.asInstanceOf[Tensor[Float]])
//      buffer.append(gradsrc_i)
//      buffer.append(gradWeight.native)
//      buffer.append(gradWeight_i.native)
//      buffer.append(gradBias.native)
//
//      updateGradInputTensors = buffer.toArray
//    }

//    updateWithNewTensor(updateGradInputTensors, 0, input)
//    updateWithNewTensor(updateGradInputTensors, 7, gradOutput)

    // TODO:
//    MklDnnOps.streamSubmit(runtime.stream, 1, updateGradInputPrimitives,
//      updateGradInputPrimitives.length,
//      updateGradInputMemoryPrimitives, updateGradInputTensors)

    gradWeight.sync()
    gradWeight_i.sync()
    gradBias.sync()

    gradInput
  }

  override def accGradParameters(input: Activity, gradOutput: Activity): Unit = {
    // Do nothing
  }

  override def parameters(): (Array[Tensor[Float]], Array[Tensor[Float]]) = {
    (Array(weight.dense, bias.dense, weight_i.dense),
      Array(gradWeight.dense, gradBias.dense, gradWeight_i.dense))
  }

  override def zeroGradParameters(): Unit = {
  }

}

object RNN{
  def apply(
   mode: Int,
   inputSize: Int,
   hiddenSize: Int,
   f: Int,
   direction: Int,
   layers: Int = 1,
   flags: Int = 0, // TODO: RNNCellFlags.RNNCellWithRelu,
   alpha: Float = 0F,
   clipping: Float = 0F,
   initWeight: Tensor[Float] = null,
   initWeightIter: Tensor[Float] = null,
   initBias: Tensor[Float] = null
 ): RNN = new RNN(mode, inputSize, hiddenSize, f, direction, layers, flags, alpha,
    clipping, initWeight, initWeightIter, initBias)
}
