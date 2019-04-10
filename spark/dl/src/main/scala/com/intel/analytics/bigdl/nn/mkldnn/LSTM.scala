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
import com.intel.analytics.bigdl.nn.VariableFormat
import com.intel.analytics.bigdl.nn.abstractnn.{Activity, Initializable}
import com.intel.analytics.bigdl.nn.mkldnn.Phase.InferencePhase
import com.intel.analytics.bigdl.tensor.Tensor

import scala.collection.mutable.ArrayBuffer

class LSTM(
  val inputSize: Int,
  val f: Int,
  val flags: Int,
  val alpha: Float,
  val clipping: Float,
  val direction: Int,
  private val initWeight: Tensor[Float] = null,
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

  private[mkldnn] val weight: TensorMMap = new TensorMMap(Array(common_n_layers,
    1, inputSize, lstm_n_gates, inputSize))
  private[mkldnn] val bias: TensorMMap = new TensorMMap(Array(common_n_layers, 1,
    lstm_n_gates, inputSize))

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
        throw new UnsupportedOperationException("Not support other input formats")
    }

    val weightShape = weight.size() /* ldigo */
    val weightLayout = Memory.Format.ldigo
    val biasShape = bias.size() /* ldgo */
    val biasLayout = Memory.Format.ldgo
    val outputShape = inputShape /* tnc */
    val outputLayout = inputLayout

    val inputShape_iter = Array(common_n_layers, 1, lstm_n_states, inputs(0).shape(1), inputSize)
    val weightShape_iter = weightShape
    val outputShape_iter = inputShape_iter

    src_layer = NativeData(inputShape, Memory.Format.any)
    src_iter = NativeData(inputShape_iter, Memory.Format.any)
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

    /*
    _inputFormats = nativeData(inputs)
    _inputFormats(0).shape.length match {
      case 3 => /* tnc */
        val t = _inputFormats(0).shape(0)
        /* Sequence length */
        val n = _inputFormats(0).shape(1)
        /* Batch Size */
        val c = _inputFormats(0).shape(2) /* Common feature size */

        direction match {
          case Direction.UnidirectionalLeft2Right =>

            src_layer_MD = _inputFormats(0).getMemoryDescription() /* tnc */

            src_iter_MD = MklDnn.MemoryDescInit(5, Array(common_n_layers, 1,
              lstm_n_states, n, c), inputs(0).dataType, Memory.Format.any) /* ldsnc */

            weights_layer_MD = MklDnn.MemoryDescInit(5, Array(common_n_layers, 1,
              c, lstm_n_gates, c), inputs(0).dataType, Memory.Format.any) /* ldigo */

            weights_iter_MD = MklDnn.MemoryDescInit(5, Array(common_n_layers, 1,
              c, lstm_n_gates, c), inputs(0).dataType, Memory.Format.any) /* ldigo */

            bias_MD = MklDnn.MemoryDescInit(4, Array(common_n_layers, 1, lstm_n_gates, c),
              inputs(0).dataType, Memory.Format.any) /* ldgo */

            dist_layer_MD = _inputFormats(0).getMemoryDescription() /* tnc */

            dist_iter_MD = MklDnn.MemoryDescInit(5, Array(common_n_layers, 1,
              lstm_n_states, n, c), inputs(0).dataType, Memory.Format.any) /* ldsnc */




          case _ => throw new UnsupportedOperationException("Not support other directions")
        }

      case _ => throw new UnsupportedOperationException("Not support other memory formats")
    }
    */
  }


  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    initMemoryDescs(inputs)
    val rnnCellDesc = MklDnn.RNNCellDescInit(AlgKind.VanillaLstm, f, flags, alpha, clipping)

    val kind = if (phase == InferencePhase) {
      PropKind.ForwardInference
    } else {
      throw new UnsupportedOperationException("Not support training")
    }

    val description = MklDnn.RNNForwardDescInit(kind, rnnCellDesc, direction, src_layer_MD,
      src_iter_MD, weights_layer_MD, weights_iter_MD, bias_MD, dist_layer_MD, dist_iter_MD)

    fwdPD = MklDnn.PrimitiveDescCreate(description, runtime.engine, 0L)

    val List(realSrc, realWei, realDst) = List(Query.SrcPd, Query.WeightsPd, Query.DstPd).map {x =>
      MemoryData.operationWant(fwdPD, x)
    }

    require(weight.size().product == realWei.shape.product,
      s"${getName} weight shape is not correct.")

    weight.setMemoryData(HeapData(weight.size(), Memory.Format.ldigo), realWei, runtime)
    bias.setMemoryData(HeapData(bias.size(), Memory.Format.ldgo),
      NativeData(bias.size(), Memory.Format.ldgo), runtime)

    weight.sync()
    bias.sync()

    val srcs = Array(realSrc.getPrimitive(runtime), realWei.getPrimitive(runtime),
      bis.getPrimitive(runtime))
    val indexes = Array.fill(srcs.length)(0)
    val dsts = Array(realDst.getPrimitive(runtime))

    updateOutputPrimitives = Array(MklDnn.PrimitiveCreate2(fwdPD, srcs, indexes, srcs.length,
      dsts, dsts.length))

    output = initTensor(realDst)

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
      weight.sync()
      bias.sync()
    }

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
    f: Int,
    flags: Int,
    alpha: Float,
    clipping: Float,
    direction: Int,
    initWeight: Tensor[Float] = null,
    initBias: Tensor[Float] = null
  ): LSTM = new LSTM(inputSize, f, flags, alpha, clipping, direction, initWeight,
    initBias)
}