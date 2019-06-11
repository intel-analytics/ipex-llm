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
import com.intel.analytics.bigdl.nn.{InitializationMethod, RandomUniform, VariableFormat, Zeros}
import com.intel.analytics.bigdl.nn.abstractnn.{Activity, Initializable}
import com.intel.analytics.bigdl.tensor.Tensor

import scala.collection.mutable.ArrayBuffer

/**
 * @param mode       : the type of RNN cell
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
  val flags: Int = RNNCellFlags.RNNCellWithRelu,
  val alpha: Float = 0F,
  val clipping: Float = 0F,
  private val initWeight: Tensor[Float] = null,
  private val initWeightIter: Tensor[Float] = null,
  private val initBias: Tensor[Float] = null
) extends MklDnnLayer with Initializable {
  private var src_layer_MD: Long = _
  private var src_iter_MD: Long = _
  private var weights_layer_MD: Long = _
  private var weights_iter_MD: Long = _
  private var bis_MD: Long = _
  private var dist_layer_MD: Long = _
  private var dist_iter_MD: Long = _

  private var fwdPD: Long = _

  private var updateOutputMemoryPrimitives: Array[Long] = _
  private var updateOutputTensors: Array[Tensor[Float]] = _

  private val common_n_layers: Int = layers
  private var ngates: Int = _
  private var nstates: Int = _

  private[mkldnn] var weight: TensorMMap = _
  private[mkldnn] var weight_i: TensorMMap = _
  private[mkldnn] var bias: TensorMMap = _
  private[mkldnn] var src_i: TensorMMap = _
  private[mkldnn] var dst_i: TensorMMap = _

  if(layers > 1) {
    require(inputSize == hiddenSize,
      "If layers of LSTM is more than 1, the input size and the hidden size should equal.\n"
      + "inputSize: " + inputSize + '\n'
      + "hiddenSize: " + hiddenSize)
  }

  mode match {
    case AlgKind.VanillaLstm =>
      ngates = 4
      nstates = 2
    case _ =>
      throw new UnsupportedOperationException("Not support such RNN Cell. Cell type: " + mode)
  }

  direction match {
    case Direction.UnidirectionalLeft2Right
         | Direction.UnidirectionalRight2Left =>

      /**
       * Gate order matching between MKLDNN LSTM and nn/LSTM:
       * MKLDNN Gate 1 -> nn/LSTM Gate 1 (input gate)
       * MKLDNN Gate 2 -> nn/LSTM Gate 3 (forget gate)
       * MKLDNN Gate 3 -> nn/LSTM Gate 2 (hidden)
       * MKLDNN Gate 4 -> nn/LSTM Gate 4 (output gate)
       */
      weight = new TensorMMap(Array(common_n_layers, 1, inputSize, ngates, hiddenSize))
      weight_i = new TensorMMap(Array(common_n_layers, 1, hiddenSize, ngates, hiddenSize))
      bias = new TensorMMap(Array(common_n_layers, 1, ngates, hiddenSize))

    case Direction.BidirectionalConcat =>
      require(layers == 1, "Bidirectional Concat LSTM does not support multiple layers. " +
        "layers = " + layers)

      weight = new TensorMMap(Array(common_n_layers, 2, inputSize, ngates, hiddenSize))
      weight_i = new TensorMMap(Array(common_n_layers, 2, hiddenSize, ngates, hiddenSize))
      bias = new TensorMMap(Array(common_n_layers, 2, ngates, hiddenSize))

    case Direction.BidirectionalSum =>

      /** TODO: Multi-layer Bidirectional LSTM is available in MKLDNN,
       * but it is not supported in current version BigDL BLAS.
       */

      weight = new TensorMMap(Array(common_n_layers, 2, inputSize, ngates, hiddenSize))
      weight_i = new TensorMMap(Array(common_n_layers, 2, hiddenSize, ngates, hiddenSize))
      bias = new TensorMMap(Array(common_n_layers, 2, ngates, hiddenSize))
  }

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

  private def initMemoryDescs(inputs: Array[MemoryData]) = {
    // TODO: The default format of input is TNC
    /**
     * The default format of input is TNC.
     * Batch size of input is needed by creating memory descriptors of src iter and dst iter.
     * Step size of input is needed by creating memory descriptor of dst layer.
     * By default, batch size of input is the second element of inputShape
     * and step size is the first element of inputShape.
     */
    val(inputShape, inputLayout) = inputs(0).layout match {
      case Memory.Format.tnc => /* tnc */
        (inputs(0).shape, Memory.Format.tnc)
      case _ =>
        throw new UnsupportedOperationException("Not support such input format. " +
          "The input format is: " + inputs(0).layout)
    }

    direction match {
      case Direction.UnidirectionalLeft2Right
           | Direction.UnidirectionalRight2Left =>
        val weightShape = weight.size() /* ldigo */
        val biasShape = bias.size() /* ldgo */
        val outputShape = Array(inputShape(0), inputShape(1), hiddenSize) /* tnc */

        val inputShape_iter = Array(common_n_layers, 1, nstates,
          inputShape(1), hiddenSize) /* ldsnc */
        val weightShape_iter = weight_i.size() /* ldigo */
        val outputShape_iter = inputShape_iter /* ldsnc */

        val src_layer = NativeData(inputShape, Memory.Format.any)
        val src_iter = NativeData(inputShape_iter, Memory.Format.any)
        val wei_layer = NativeData(weightShape, Memory.Format.any)
        val wei_iter = NativeData(weightShape_iter, Memory.Format.any)
        val bis = NativeData(biasShape, Memory.Format.any)
        val dst_layer = NativeData(outputShape, Memory.Format.any)
        val dst_iter = NativeData(outputShape_iter, Memory.Format.any)

        src_layer_MD = src_layer.getMemoryDescription()
        src_iter_MD = src_iter.getMemoryDescription()
        weights_layer_MD = wei_layer.getMemoryDescription()
        weights_iter_MD = wei_iter.getMemoryDescription()
        bis_MD = bis.getMemoryDescription()
        dist_layer_MD = dst_layer.getMemoryDescription()
        dist_iter_MD = dst_iter.getMemoryDescription()

        src_i = new TensorMMap(inputShape_iter)
        src_i.dense.copy(Tensor[Float]().resize(inputShape_iter).zero())
        dst_i = new TensorMMap(outputShape_iter)
        dst_i.dense.copy(Tensor[Float]().resize(outputShape_iter).zero())

      case Direction.BidirectionalConcat =>
        val weightShape = weight.size() /* ldigo */
        val biasShape = bias.size() /* ldgo */
        val outputShape = Array(inputShape(0), inputShape(1), 2 * hiddenSize) /* tnc */

        val inputShape_iter = Array(common_n_layers, 2, nstates,
          inputShape(1), hiddenSize) /* ldsnc */
        val weightShape_iter = weight_i.size() /* ldigo */
        val outputShape_iter = inputShape_iter /* ldsnc */

        val src_layer = NativeData(inputShape, Memory.Format.any)
        val src_iter = NativeData(inputShape_iter, Memory.Format.any)
        val wei_layer = NativeData(weightShape, Memory.Format.any)
        val wei_iter = NativeData(weightShape_iter, Memory.Format.any)
        val bis = NativeData(biasShape, Memory.Format.any)
        val dst_layer = NativeData(outputShape, Memory.Format.any)
        val dst_iter = NativeData(outputShape_iter, Memory.Format.any)

        src_layer_MD = src_layer.getMemoryDescription()
        src_iter_MD = src_iter.getMemoryDescription()
        weights_layer_MD = wei_layer.getMemoryDescription()
        weights_iter_MD = wei_iter.getMemoryDescription()
        bis_MD = bis.getMemoryDescription()
        dist_layer_MD = dst_layer.getMemoryDescription()
        dist_iter_MD = dst_iter.getMemoryDescription()

        /** TODO: user-defined initial hidden state is not supported currently.
         * The default initial hidden state is all zero.
         */
        src_i = new TensorMMap(inputShape_iter)
        src_i.dense.copy(Tensor[Float]().resize(inputShape_iter).zero())
        dst_i = new TensorMMap(outputShape_iter)
        dst_i.dense.copy(Tensor[Float]().resize(outputShape_iter).zero())

      case Direction.BidirectionalSum =>
        val weightShape = weight.size() /* ldigo */
        val biasShape = bias.size() /* ldgo */
        val outputShape = Array(inputShape(0), inputShape(1), hiddenSize) /* tnc */

        val inputShape_iter = Array(common_n_layers, 2, nstates,
          inputShape(1), hiddenSize) /* ldsnc */
        val weightShape_iter = weight_i.size() /* ldigo */
        val outputShape_iter = inputShape_iter /* ldsnc */

        val src_layer = NativeData(inputShape, Memory.Format.any)
        val src_iter = NativeData(inputShape_iter, Memory.Format.any)
        val wei_layer = NativeData(weightShape, Memory.Format.any)
        val wei_iter = NativeData(weightShape_iter, Memory.Format.any)
        val bis = NativeData(biasShape, Memory.Format.any)
        val dst_layer = NativeData(outputShape, Memory.Format.any)
        val dst_iter = NativeData(outputShape_iter, Memory.Format.any)

        src_layer_MD = src_layer.getMemoryDescription()
        src_iter_MD = src_iter.getMemoryDescription()
        weights_layer_MD = wei_layer.getMemoryDescription()
        weights_iter_MD = wei_iter.getMemoryDescription()
        bis_MD = bis.getMemoryDescription()
        dist_layer_MD = dst_layer.getMemoryDescription()
        dist_iter_MD = dst_iter.getMemoryDescription()

        src_i = new TensorMMap(inputShape_iter)
        src_i.dense.copy(Tensor[Float]().resize(inputShape_iter).zero())
        dst_i = new TensorMMap(outputShape_iter)
        dst_i.dense.copy(Tensor[Float]().resize(outputShape_iter).zero())

      case _ => throw new UnsupportedOperationException("Not support such direction")
    }
  }

  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    val kind = if (!isTraining()) {
      PropKind.ForwardInference
    } else {
      throw new UnsupportedOperationException("Not support training")
    }

    val rnnCellDesc = mode match {
      case AlgKind.VanillaLstm =>
        MklDnn.RNNCellDescInit(AlgKind.VanillaLstm, f, flags, alpha, clipping)
      case _ => throw new UnsupportedOperationException("Not support such RNN cell. " +
        "Cell type: " + mode)
    }

    initMemoryDescs(inputs)

    val description = MklDnn.RNNForwardDescInit(kind, rnnCellDesc, direction, src_layer_MD,
      src_iter_MD, weights_layer_MD, weights_iter_MD, bis_MD, dist_layer_MD, dist_iter_MD)

    fwdPD = MklDnn.PrimitiveDescCreate(description, runtime.engine, 0L)

    val realSrc = MemoryData.operationWant(fwdPD, Query.SrcPd, 0)
    val realSrc_iter = MemoryData.operationWant(fwdPD, Query.SrcPd, 1)
    val realWei = MemoryData.operationWant(fwdPD, Query.WeightsPd, 0)
    val realWei_iter = MemoryData.operationWant(fwdPD, Query.WeightsPd, 1)
    val realBias = MemoryData.operationWant(fwdPD, Query.WeightsPd, 2)

    val realDst = MemoryData.operationWant(fwdPD, Query.DstPd, 0)
    val realDst_iter = MemoryData.operationWant(fwdPD, Query.DstPd, 1)

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
    throw new UnsupportedOperationException("Not support backward propagation")
  }

  override def parameters(): (Array[Tensor[Float]], Array[Tensor[Float]]) = {
    (Array(weight.dense, bias.dense, weight_i.dense), Array())
  }

  override def zeroGradParameters(): Unit = {
  }

  override def release(): Unit = {
    super.release()
    List(weight, bias, weight_i, src_i, dst_i).foreach(_.release())
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
   flags: Int = RNNCellFlags.RNNCellWithRelu,
   alpha: Float = 0F,
   clipping: Float = 0F,
   initWeight: Tensor[Float] = null,
   initWeightIter: Tensor[Float] = null,
   initBias: Tensor[Float] = null
 ): RNN = new RNN(mode, inputSize, hiddenSize, f, direction, layers, flags, alpha,
    clipping, initWeight, initWeightIter, initBias)
}
