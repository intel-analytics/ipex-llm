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
import com.intel.analytics.bigdl.nn.{InitializationMethod, RandomUniform, VariableFormat}
import com.intel.analytics.bigdl.nn.abstractnn.{Activity, Initializable}
import com.intel.analytics.bigdl.nn.mkldnn.Phase.InferencePhase
import com.intel.analytics.bigdl.tensor.Tensor

import scala.collection.mutable.ArrayBuffer

class LSTM(
  val inputSize: Int,
  val hiddenSize: Int,
  val f: Int,
  val direction: Int,
  val flags: Int = RNNCellFlags.RNNCellWithRelu,
  val alpha: Float = 0F,
  val clipping: Float = 0F,
  private val initWeight: Tensor[Float] = null,
  private val initWeightIter: Tensor[Float] = null,
  private val initBias: Tensor[Float] = null
) extends MklDnnLayer with Initializable {
  @transient private var src_layer_MD: Long = _
  @transient private var src_iter_MD: Long = _
  @transient private var weights_layer_MD: Long = _
  @transient private var weights_iter_MD: Long = _
  @transient private var bias_MD: Long = _
  @transient private var dist_layer_MD: Long = _
  @transient private var dist_iter_MD: Long = _

  @transient private var fwdPD: Long = _

  @transient private var updateOutputMemoryPrimitives: Array[Long] = _
  @transient private var updateOutputTensors: Array[Tensor[Float]] = _

  private val common_n_layers: Int = 1
  private val lstm_n_gates: Int = 4
  private val lstm_n_states: Int = 2

  private[mkldnn] var weight: TensorMMap = _
  private[mkldnn] var weight_i: TensorMMap = _
  private[mkldnn] var bias: TensorMMap = _
  private[mkldnn] var src_i: TensorMMap = _
  private[mkldnn] var dst_i: TensorMMap = _

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

  private var src_layer: NativeData = _
  private var src_iter: NativeData = _
  private var wei_layer: NativeData = _
  private var wei_iter: NativeData = _
  private var bis: NativeData = _
  private var dst: NativeData = _
  private var dst_iter: NativeData = _

  private[mkldnn] def initMemoryDescs(inputs: Array[MemoryData]) = {
    val(inputShape, inputLayout) = inputs(0).shape.length match {
      case 3 => /* tnc */
        (inputs(0).shape, Memory.Format.tnc)
      case _ =>
        throw new UnsupportedOperationException("Not support such input format")
    }

    direction match {
      case Direction.UnidirectionalLeft2Right
           | Direction.UnidirectionalRight2Left =>
        weight = new TensorMMap(Array(common_n_layers, 1, inputSize, lstm_n_gates, hiddenSize))
        weight_i = new TensorMMap(Array(common_n_layers, 1, hiddenSize, lstm_n_gates, hiddenSize))
        bias = new TensorMMap(Array(common_n_layers, 1, lstm_n_gates, hiddenSize))

        {
          val stdv = 1.0 / math.sqrt(hiddenSize)
          val wInit: InitializationMethod = RandomUniform(-stdv, stdv)
          val bInit: InitializationMethod = RandomUniform(-stdv, stdv)
          setInitMethod(wInit, bInit)
        }

        val weightShape = weight.size() /* ldigo */
        val biasShape = bias.size() /* ldgo */
        val outputShape = Array(inputShape(0), inputShape(1), hiddenSize) /* tnc */

        val inputShape_iter = Array(common_n_layers, 1, lstm_n_states,
            inputs(0).shape(1), hiddenSize) /* ldsnc */
        val weightShape_iter = weight_i.size() /* ldigo */
        val outputShape_iter = inputShape_iter /* ldsnc */

        src_layer = NativeData(inputShape, Memory.Format.any)
        src_iter = NativeData(inputShape_iter, Memory.Format.any)  // TODO any or ldsnc?
        /* TODO Refer to MKLDNN details, Format of src_iter cannot be any */
        wei_layer = NativeData(weightShape, Memory.Format.any)
        wei_iter = NativeData(weightShape_iter, Memory.Format.any)
        bis = NativeData(biasShape, Memory.Format.any)
        dst = NativeData(outputShape, Memory.Format.any)
        dst_iter = NativeData(outputShape_iter, Memory.Format.any)

        src_layer_MD = src_layer.getMemoryDescription()
        src_iter_MD = src_iter.getMemoryDescription()
        weights_layer_MD = wei_layer.getMemoryDescription()
        weights_iter_MD = wei_iter.getMemoryDescription()
        bias_MD = bis.getMemoryDescription()
        dist_layer_MD = dst.getMemoryDescription()
        dist_iter_MD = dst_iter.getMemoryDescription()

        src_i = new TensorMMap(inputShape_iter)
        src_i.dense.copy(Tensor[Float]().resize(inputShape_iter).zero())
        dst_i = new TensorMMap(outputShape_iter)
        dst_i.dense.copy(Tensor[Float]().resize(outputShape_iter).zero())

      case Direction.BidirectionalConcat =>
        weight = new TensorMMap(Array(common_n_layers, 2, inputSize, lstm_n_gates, hiddenSize))
        weight_i = new TensorMMap(Array(common_n_layers, 2, hiddenSize, lstm_n_gates, hiddenSize))
        bias = new TensorMMap(Array(common_n_layers, 2, lstm_n_gates, hiddenSize))

        {
          val stdv = 1.0 / math.sqrt(hiddenSize)
          val wInit: InitializationMethod = RandomUniform(-stdv, stdv)
          val bInit: InitializationMethod = RandomUniform(-stdv, stdv)
          setInitMethod(wInit, bInit)
        }

        val weightShape = weight.size() /* ldigo */
        val biasShape = bias.size() /* ldgo */
        val outputShape = Array(inputShape(0), inputShape(1), 2 * hiddenSize) /* tnc */

        val inputShape_iter = Array(common_n_layers, 2, lstm_n_states,
            inputs(0).shape(1), hiddenSize) /* ldsnc */
        val weightShape_iter = weight_i.size() /* ldigo */
        val outputShape_iter = inputShape_iter /* ldsnc */

        src_layer = NativeData(inputShape, Memory.Format.any)
        src_iter = NativeData(inputShape_iter, Memory.Format.any)  // TODO any or ldsnc?
        /* TODO Refer to MKLDNN details, Format of src_iter cannot be any */
        wei_layer = NativeData(weightShape, Memory.Format.any)
        wei_iter = NativeData(weightShape_iter, Memory.Format.any)
        bis = NativeData(biasShape, Memory.Format.any)
        dst = NativeData(outputShape, Memory.Format.any)
        dst_iter = NativeData(outputShape_iter, Memory.Format.any)

        src_layer_MD = src_layer.getMemoryDescription()
        src_iter_MD = src_iter.getMemoryDescription()
        weights_layer_MD = wei_layer.getMemoryDescription()
        weights_iter_MD = wei_iter.getMemoryDescription()
        bias_MD = bis.getMemoryDescription()
        dist_layer_MD = dst.getMemoryDescription()
        dist_iter_MD = dst_iter.getMemoryDescription()

        src_i = new TensorMMap(inputShape_iter)
        src_i.dense.copy(Tensor[Float]().resize(inputShape_iter).zero())
        dst_i = new TensorMMap(outputShape_iter)
        dst_i.dense.copy(Tensor[Float]().resize(outputShape_iter).zero())

      case Direction.BidirectionalSum =>
        weight = new TensorMMap(Array(common_n_layers, 2, inputSize, lstm_n_gates, hiddenSize))
        weight_i = new TensorMMap(Array(common_n_layers, 2, hiddenSize, lstm_n_gates, hiddenSize))
        bias = new TensorMMap(Array(common_n_layers, 2, lstm_n_gates, hiddenSize))

        {
          val stdv = 1.0 / math.sqrt(hiddenSize)
          val wInit: InitializationMethod = RandomUniform(-stdv, stdv)
          val bInit: InitializationMethod = RandomUniform(-stdv, stdv)
          setInitMethod(wInit, bInit)
        }

        val weightShape = weight.size() /* ldigo */
        val biasShape = bias.size() /* ldgo */
        val outputShape = Array(inputShape(0), inputShape(1), hiddenSize) /* tnc */

        val inputShape_iter = Array(common_n_layers, 2, lstm_n_states,
          inputs(0).shape(1), hiddenSize) /* ldsnc */
        val weightShape_iter = weight_i.size() /* ldigo */
        val outputShape_iter = inputShape_iter /* ldsnc */

        src_layer = NativeData(inputShape, Memory.Format.any)
        src_iter = NativeData(inputShape_iter, Memory.Format.any)  // TODO any or ldsnc?
        /* TODO Refer to MKLDNN details, Format of src_iter cannot be any */
        wei_layer = NativeData(weightShape, Memory.Format.any)
        wei_iter = NativeData(weightShape_iter, Memory.Format.any)
        bis = NativeData(biasShape, Memory.Format.any)
        dst = NativeData(outputShape, Memory.Format.any)
        dst_iter = NativeData(outputShape_iter, Memory.Format.any)

        src_layer_MD = src_layer.getMemoryDescription()
        src_iter_MD = src_iter.getMemoryDescription()
        weights_layer_MD = wei_layer.getMemoryDescription()
        weights_iter_MD = wei_iter.getMemoryDescription()
        bias_MD = bis.getMemoryDescription()
        dist_layer_MD = dst.getMemoryDescription()
        dist_iter_MD = dst_iter.getMemoryDescription()

        src_i = new TensorMMap(inputShape_iter)
        src_i.dense.copy(Tensor[Float]().resize(inputShape_iter).zero())
        dst_i = new TensorMMap(outputShape_iter)
        dst_i.dense.copy(Tensor[Float]().resize(outputShape_iter).zero())

      case _ => throw new UnsupportedOperationException("Not support such direction")
    }
  }

  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    val rnnCellDesc = MklDnn.RNNCellDescInit(AlgKind.VanillaLstm, f, flags, alpha, clipping)

    val kind = if (phase == InferencePhase) {
      PropKind.ForwardInference
    } else {
      throw new UnsupportedOperationException("Not support training")
    }

    val description = MklDnn.RNNForwardDescInit(kind, rnnCellDesc, direction, src_layer_MD,
      src_iter_MD, weights_layer_MD, weights_iter_MD, bias_MD, dist_layer_MD, dist_iter_MD)

    fwdPD = MklDnn.PrimitiveDescCreate(description, runtime.engine, 0L)

    val realSrc = MemoryData.operationWant2(fwdPD, Query.SrcPd, 0)
    val realSrc_iter = MemoryData.operationWant2(fwdPD, Query.SrcPd, 1)
    val realWei = MemoryData.operationWant2(fwdPD, Query.WeightsPd, 0)
    val realWei_iter = MemoryData.operationWant2(fwdPD, Query.WeightsPd, 1)
    val realBias = MemoryData.operationWant2(fwdPD, Query.WeightsPd, 2)

    val realDst = MemoryData.operationWant2(fwdPD, Query.DstPd, 0)
    val realDst_iter = MemoryData.operationWant2(fwdPD, Query.DstPd, 1)

    require(src_i.size().product == realSrc_iter.shape.product,
      s"${getName} src iter shape is not correct.")
    require(dst_i.size().product == realDst_iter.shape.product,
      s"${getName} dst iter shape is not correct.")
    require(weight.size().product == realWei.shape.product,
      s"${getName} weight shape is not correct.")
    require(weight_i.size().product == realWei_iter.shape.product,
      s"${getName} weight iter shape is not correct.")
    require(bias.size().product == realBias.shape.product,
      s"${getName} bias shape is not correct.")

    weight.setMemoryData(HeapData(weight.size(), Memory.Format.ldigo), realWei, runtime)
    weight_i.setMemoryData(HeapData(weight_i.size(), Memory.Format.ldigo), realWei_iter, runtime)
    bias.setMemoryData(HeapData(bias.size(), Memory.Format.ldgo), realBias, runtime)
    src_i.setMemoryData(HeapData(src_i.size(), Memory.Format.ldsnc), realSrc_iter, runtime)
    dst_i.setMemoryData(HeapData(dst_i.size(), Memory.Format.ldsnc), realDst_iter, runtime)

    weight.sync()
    weight_i.sync()
    bias.sync()
    src_i.sync()
    dst_i.sync()

    val srcs = Array(realSrc.getPrimitive(runtime), realSrc_iter.getPrimitive(runtime),
      realWei.getPrimitive(runtime), realWei_iter.getPrimitive(runtime),
      realBias.getPrimitive(runtime))
    val indexes = Array.fill(srcs.length)(0)

    val dsts = Array(realDst.getPrimitive(runtime), realDst_iter.getPrimitive(runtime))

    val primitive = MklDnn.PrimitiveCreate2(fwdPD, srcs, indexes, srcs.length, dsts, dsts.length)

    updateOutputMemoryPrimitives = srcs ++ dsts
    updateOutputPrimitives = Array(primitive)
    output = initTensor(realDst)

    _inputFormats = Array(realSrc)
    _outputFormats = Array(realDst)

    (_inputFormats, _outputFormats)
  }

  override def updateOutput(input: Activity): Activity = {
    if (updateOutputTensors == null) {
      val buffer = new ArrayBuffer[Tensor[Float]]()
      buffer.append(input.asInstanceOf[Tensor[Float]])
      buffer.append(src_i.native)
      buffer.append(weight.native)
      buffer.append(weight_i.native)
      buffer.append(bias.native)
      buffer.append(output.asInstanceOf[Tensor[Float]])
      buffer.append(dst_i.native)

      updateOutputTensors = buffer.toArray
    }

    updateWithNewTensor(updateOutputTensors, 0, input)

    MklDnnOps.streamSubmit(runtime.stream, 1, updateOutputPrimitives, updateOutputPrimitives.length,
      updateOutputMemoryPrimitives, updateOutputTensors)

    output
  }

  override private[mkldnn] def initBwdPrimitives(grad: Array[MemoryData], phase: Phase) = {
    _gradInputFormats = grad.clone()
    _gradOutputFormats = grad.clone()
    (_gradInputFormats, _gradOutputFormats)
  }
}

object LSTM{
  def apply(
   inputSize: Int,
   hiddenSize: Int,
   f: Int,
   direction: Int,
   flags: Int = RNNCellFlags.RNNCellWithRelu,
   alpha: Float = 0F,
   clipping: Float = 0F,
   initWeight: Tensor[Float] = null,
   initWeightIter: Tensor[Float] = null,
   initBias: Tensor[Float] = null
 ): LSTM = new LSTM(inputSize, hiddenSize, f, direction, flags, alpha,
    clipping, initWeight, initWeightIter, initBias)
}
