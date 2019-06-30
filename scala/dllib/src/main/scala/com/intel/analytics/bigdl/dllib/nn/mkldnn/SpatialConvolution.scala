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

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.mkl._
import com.intel.analytics.bigdl.nn.{Utils => NNUtils, _}
import com.intel.analytics.bigdl.nn.abstractnn._
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.{DenseTensorMath, DnnTensor, Tensor}

import scala.collection.mutable.ArrayBuffer

/**
 * Applies a 2D convolution over an input image composed of several input planes.
 * The input tensor in forward(input) is expected to be
 * a 3D tensor (nInputPlane x height x width).
 *
 * nInputPlane The number of expected input planes in the image given into forward()
 * nOutputPlane: The number of output planes the convolution layer will produce.
 * kernelW: the kernel width of the convolution
 * kernelH: The kernel height of the convolution
 * strideW: Int = 1, The step of the convolution in the width dimension.
 * strideH: Int = 1, The step of the convolution in the height dimension
 * padW: Int = 0, The additional zeros added per width to the input planes.
 * padH: Int = 0, The additional zeros added per height to the input planes.
 * nGroup: Int = 1, Kernel group number
 * propagateBack: Boolean = true, propagate gradient back
 * wRegularizer: Regularizer[Float] = null,
 * bRegularizer: Regularizer[Float] = null,
 * initWeight: Tensor[Float] = null,
 * initBias: Tensor[Float] = null,
 * initGradWeight: Tensor[Float] = null,
 * initGradBias: Tensor[Float] = null,
 * withBias: Boolean = true,
 * format: DataFormat = DataFormat.NCHW,
 * dilationW: Int = 1,
 * dilationH: Int = 1
 *
 * When padW and padH are both -1, we use a padding algorithm similar to the "SAME"
 * padding of tensorflow. That is
 *
 * outHeight = Math.ceil(inHeight.toFloat/strideH.toFloat)
 * outWidth = Math.ceil(inWidth.toFloat/strideW.toFloat)
 *
 * padAlongHeight = Math.max(0, (outHeight - 1) * strideH + kernelH - inHeight)
 * padAlongWidth = Math.max(0, (outWidth - 1) * strideW + kernelW - inWidth)
 *
 * padTop = padAlongHeight / 2
 * padLeft = padAlongWidth / 2
 *
 * @param wRegularizer: instance of [[Regularizer]]
 *                    (eg. L1 or L2 regularization), applied to the input weights matrices.
 * @param bRegularizer: instance of [[Regularizer]]
 *                    applied to the bias.
 * @param dilationW: dilation width, default to 1
 * @param dilationH: dilation height, default to 1
 */
class SpatialConvolution(
  val nInputPlane: Int,
  val nOutputPlane: Int,
  val kernelW: Int,
  val kernelH: Int,
  val strideW: Int = 1,
  val strideH: Int = 1,
  val padW: Int = 0,
  val padH: Int = 0,
  val nGroup: Int = 1,
  val propagateBack: Boolean = true,
  var wRegularizer: Regularizer[Float] = null,
  var bRegularizer: Regularizer[Float] = null,
  val initWeight: Tensor[Float] = null,
  val initBias: Tensor[Float] = null,
  val initGradWeight: Tensor[Float] = null,
  val initGradBias: Tensor[Float] = null,
  val withBias: Boolean = true,
  val format: DataFormat = DataFormat.NCHW,
  val dilationW: Int = 1,
  val dilationH: Int = 1
) extends MklDnnLayer with Initializable with Serializable with MklInt8Convertible {
  require(nInputPlane % nGroup == 0, s"Number of input channels " +
    s"should be multiples of group " +
    s"number of input channels ${nInputPlane}, " +
    s"group ${nGroup}.")
  require(nOutputPlane % nGroup == 0,
    "Number of output channels " +
      "should be multiples of group " +
      s"(number of output channels ${nOutputPlane}, " +
      s"group ${nGroup}).")

  private val weightShape = if (nGroup == 1) {
    Array(nOutputPlane, nInputPlane, kernelH, kernelW)
  } else {
    Array (nGroup, nOutputPlane / nGroup, nInputPlane / nGroup, kernelH, kernelW)
  }

  // !!!important!!! this is for weight and input conversion.
  // The weight in forward and updateGradInput is different.
  // The input in updateOutput and accGradParameters is different too.
  // It's `lazy` so the reordermanager need not serialized.
  @transient private lazy val reorderManager = new ReorderManager

  private[mkldnn] val weight = new TensorMMap(weightShape)
  private[mkldnn] val bias = new TensorMMap(Array(nOutputPlane))
  private[mkldnn] val gradWeight = new TensorMMap(weightShape)
  private[mkldnn] val gradBias = new TensorMMap(Array(nOutputPlane))

  // The weight maybe have different format between updateOutput and updateGradInput
  private var weightForBackward: DnnTensor[Float] = _
  private var weightForBackwardMemoryData: MemoryData = _

  // The input maybe have different format between updateOutput and accGradParameters
  private var inputForAcc: DnnTensor[Float] = _
  private var inputForAccMemoryData: MemoryData = _

  @transient private var forwardPrimDesc: Long = 0L

  @transient private var updateOutputMemoryPrimitives: Array[Long] = _
  @transient private var updateOutputTensors: Array[Tensor[Float]] = _
  @transient private var updateGradInputMemoryPrimitives: Array[Long] = _
  @transient private var updateGradInputTensors: Array[Tensor[Float]] = _
  @transient private var updateGradWMemoryPrimitives: Array[Long] = _
  @transient private var updateGradWTensors: Array[Tensor[Float]] = _
  @transient private var paddingTL: Array[Int] = _
  @transient private var paddingBR: Array[Int] = _

  var needQuantize: Boolean = false
  var negativeInput: Boolean = true

  private var _relu = false
  private var _sum = false
  private var _batchNorm = false
  private var _dim = 1
  private var _sumInput = false


  def relu: Boolean = _relu
  def setReLU(value: Boolean = true): this.type = {
    _relu = value
    this
  }

  def batchNorm: Boolean = _batchNorm
  def setBatchNorm(value: Boolean = true): this.type = {
    _batchNorm = value
    this
  }

  def sum: Boolean = _sum
  def setSum(value: Boolean = true): this.type = {
    _sum = value
    this
  }

  var sumOp: MklDnnLayer = null
  def setSumOp(conv: Module[Float], number: Int = 1): this.type = {
    sumOp = conv.asInstanceOf[MklDnnLayer]
    _dim = number
    _sum = true
    this
  }

  // get padding type
  private val paddingType: PaddingType.Value = getPaddingType()

  /*
  Parameters for Dilated Convolution
  In most deep learning frameworks,
  the default value of dilation which defines regular convolution module is 1
  BigDL follows this convention
  However the default value used by mkl-dnn is 0;
  in order to keep consistensy, we internally transform them with deducting by 1.
  */
  private val dilationW_mkldnn: Int = dilationW - 1
  private val dilationH_mkldnn: Int = dilationH - 1



  private def getOutputShape(oh: Int, ow: Int, batchSize: Int = -1): Array[Int] = {
    format match {
      case DataFormat.NCHW =>
        if (batchSize == -1) {
          Array(nOutputPlane, oh, ow)
        } else {
          Array(batchSize, nOutputPlane, oh, ow)
        }
      case DataFormat.NHWC =>
        if (batchSize == -1) {
          Array(oh, ow, nOutputPlane)
        } else {
          Array(batchSize, oh, ow, nOutputPlane)
        }

    }
  }

  {
    val stdv = 1.0 / math.sqrt(kernelW * kernelH * nInputPlane)
    val wInit: InitializationMethod = RandomUniform(-stdv, stdv)
    val bInit: InitializationMethod = if (withBias) RandomUniform(-stdv, stdv)
    else null
    setInitMethod(wInit, bInit)
  }

  override def reset(): Unit = {
    if (initWeight == null) { // TODO only support oihw format weights
      weightInitMethod.init(weight.dense, if (nGroup == 1) {
        VariableFormat.OUT_IN_KW_KH
      } else {
        VariableFormat.GP_OUT_IN_KW_KH
      })
    } else {
      weight.dense.copy(initWeight)
    }

    if (initBias == null) {
      biasInitMethod.init(bias.dense, VariableFormat.ONE_D)
    } else {
      bias.dense.copy(initBias)
    }
  }

  private def setScalesOutForAttr(scaleIn: Array[Float], scaleOut: Array[Float],
    attr: Long): Unit = {
    require(this.getWeightScales() != null, s"you should use a model contains scales")
    val scales = this.getWeightScales().flatten.map(w =>
      if (Math.abs(w - 0.0f) < DenseTensorMath.floatEpsilon) {
        0.0f
      } else {
        scaleOut(0) / (scaleIn(0) * 127.0f / w)
      }).toArray
    MklDnn.AttrSetOutputScales(attr, scales.length, 2, scales)
    MklDnn.AttrSetIntOutputRoundMode(attr, 1)
  }

  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    reorderManager.setRuntime(runtime)

    // maybe the model has no scales, we can't do quantize here.
    if (getInputScales().flatten.isEmpty || getOutputScales().flatten.isEmpty ||
      getWeightScales().flatten.isEmpty) {
      needQuantize = false
    }

    if (_sum && inputs.length > 1) {
      _sumInput = true
      require(inputs.length == 2,
        s"inputs length should be 2 when having sum operation, but get ${inputs.length}")
    }
    // we should not use output branch
    val inputMemoryData = inputs(inputs.length - _dim)
    val inputHeight = inputMemoryData.shape(2) // TODO only supports 4-D and nchw
    val inputWidth = inputMemoryData.shape(3)

    val convPaddingShape = getConvPaddingShape(inputHeight, inputWidth, paddingType)
    val convOutputShape = getConvOutputShape(inputHeight, inputWidth, convPaddingShape)

    val padTop = convPaddingShape.top
    val padBottom = convPaddingShape.bottom
    val padLeft = convPaddingShape.left
    val padRight = convPaddingShape.right
    val outputHeight = convOutputShape.height
    val outputWidth = convOutputShape.width

    paddingTL = Array(padTop, padLeft)
    paddingBR = Array(padBottom, padRight)

    val inputShape = inputMemoryData.shape
    val outputShape = Array(inputMemoryData.shape(0), nOutputPlane, outputHeight, outputWidth)

    val inputDataType = if (needQuantize) {
      if (negativeInput) {
        DataType.S8
      } else {
        DataType.U8
      }
    } else {
      DataType.F32
    }
    val weightDataType = if (needQuantize) DataType.S8 else DataType.F32
    val biasDataType = if (needQuantize) DataType.S32 else DataType.F32
    val outputDataType = if (needQuantize) {
      // must use the same datatype with the sumOp, otherwise the result will be wrong.
      if (!relu || (sum && sumOp.outputFormats()(0).dataType == DataType.S8)) {
        DataType.S8
      } else {
        DataType.U8
      }
    } else {
      DataType.F32
    }

    val src = NativeData(inputShape, Memory.Format.any, inputDataType)
    val wei = NativeData(weightShape, Memory.Format.any, weightDataType)
    val bis = NativeData(Array(nOutputPlane), Memory.Format.x, biasDataType)
    val dst = NativeData(outputShape, Memory.Format.any, outputDataType)

    val scaleIn = this.getInputScales().flatten.map { x =>
      if (negativeInput) {
        Scale.S8_MAX / x
      } else {
        Scale.U8_MAX / x
      }
    }

    val scaleOut = this.getOutputScales().flatten.map { x =>
      if (relu) {
        Scale.U8_MAX / x
      } else {
        Scale.S8_MAX / x
      }
    }

    val scaleWeight = this.getWeightScales().flatten.map { w => Scale.S8_MAX / w }

    // TODO check wether ForwardInference and ForwardTraining is the same
    val desc = MklDnn.DilatedConvForwardDescInit(
      PropKind.ForwardTraining, AlgKind.ConvolutionDirect,
      src.getMemoryDescription(),
      wei.getMemoryDescription(),
      bis.getMemoryDescription(),
      dst.getMemoryDescription(),
      Array(strideW, strideH), Array(dilationW_mkldnn, dilationH_mkldnn),
      paddingTL, paddingBR,
      MklDnn.PaddingKind.mkldnnPaddingZero)

    forwardPrimDesc = if (relu || sum) {
      val attr = MklDnn.CreateAttr()

      // create output scales for s8/u8 output
      if (needQuantize) {
        setScalesOutForAttr(scaleIn, scaleOut, attr)
      }

      val postOps = MklDnn.CreatePostOps()
      if (sum) {
        val sumScale = if (needQuantize) {
          require(scaleOut.length == sumOp.outputFormats()(0).scales.length,
            s"the output scales should be the same between ${getName()} and ${sumOp.getName()}")
          scaleOut(0) / sumOp.outputFormats()(0).scales(0)
        } else {
          1.0f
        }
        MklDnn.PostOpsAppendSum(postOps, sumScale)
      }

      if (relu) {
        MklDnn.PostOpsAppendEltwise(postOps, 1.0f, AlgKind.EltwiseRelu, 0.0f, 0.0f)
      }
      MklDnn.AttrSetPostOps(attr, postOps)

      MklDnn.PrimitiveDescCreateV2(desc, attr, runtime.engine, 0)
      // TODO we should destroy these ops
    } else if (needQuantize) {
      val attr = MklDnn.CreateAttr()

      setScalesOutForAttr(scaleIn, scaleOut, attr)

      MklDnn.PrimitiveDescCreateV2(desc, attr, runtime.engine, 0)
      // TODO we should destroy these ops
    } else {
      MklDnn.PrimitiveDescCreate(desc, runtime.engine, 0)
    }

    val List(realSrc, realWei, realDst) = List(Query.SrcPd, Query.WeightsPd, Query.DstPd).map {x =>
      MemoryData.operationWant(forwardPrimDesc, x)
    }

    // The weight on heap should be oihw or goihw format and should reorder it first when using.
    val defaultWeightLayout = if (nGroup == 1) {
      Memory.Format.oihw
    } else {
      Memory.Format.goihw
    }

    val defaultWeight = HeapData(weight.dense.size(), defaultWeightLayout)
    val defaultBias = HeapData(bias.dense.size(), Memory.Format.x)

    if (needQuantize) {
      defaultWeight.setMask(getWeightDimMask())
      defaultWeight.setScales(scaleWeight)

      defaultBias.setMask(getWeightDimMask())
      defaultBias.setScales(scaleWeight.map(w => w * scaleIn(0)))
    }

    weight.setMemoryData(defaultWeight, realWei, runtime)
    bias.setMemoryData(defaultBias, bis, runtime)

    weight.sync()
    bias.sync()

    val srcs = Array(realSrc.getPrimitive(runtime), realWei.getPrimitive(runtime),
      bis.getPrimitive(runtime))
    val indexes = Array.fill(srcs.length)(0)
    val dsts = Array(realDst.getPrimitive(runtime))

    val primitive = MklDnn.PrimitiveCreate2(forwardPrimDesc, srcs, indexes, srcs.length,
      dsts, dsts.length)

    updateOutputMemoryPrimitives = srcs ++ dsts
    updateOutputPrimitives = Array(primitive)
    output = initTensor(realDst)

    // quantize weight from fp32 to int8
    if (needQuantize) {
      realSrc.setMask(this.getInputDimMask())
      realSrc.setScales(scaleIn)
    }

    if (needQuantize) {
      realDst.setMask(this.getOutputDimMask())
      realDst.setScales(scaleOut)
    }

    _inputFormats = if (_sumInput) Array(realSrc, realSrc) else Array(realSrc)
    _outputFormats = Array(realDst)
    (_inputFormats, _outputFormats)
  }

  override def updateOutput(input: Activity): Activity = {
    val inputTensor = if (input.isTensor) {
      input.toTensor[Float]
    } else {
      // here we should not use the output branch
      output = input.toTable.get[Tensor[Float]](_dim).get
      input.toTable.get[Tensor[Float]](3 - _dim).get
    }
    if (updateOutputTensors == null) {
      val buffer = new ArrayBuffer[Tensor[Float]]()
      buffer.append(inputTensor.asInstanceOf[Tensor[Float]])
      buffer.append(weight.native)
      buffer.append(bias.native)
      if (sum && input.isTensor) {
        output = sumOp.output
      }
      buffer.append(output.asInstanceOf[Tensor[Float]])
      updateOutputTensors = buffer.toArray
    }

    updateWithNewTensor(updateOutputTensors, 0, inputTensor)

    if (isTraining()) {
      weight.sync()
      bias.sync()
    }

    MklDnnOps.streamSubmit(runtime.stream, 1, updateOutputPrimitives, updateOutputPrimitives.length,
      updateOutputMemoryPrimitives, updateOutputTensors)

    output
  }

  override private[mkldnn] def initBwdPrimitives(grad: Array[MemoryData], phase: Phase) = {
    val inputShape = inputFormats()(0).shape.length match {
      case 1 => inputFormats()(0).shape ++ Array(1) // TODO Test
      case _ => inputFormats()(0).shape
    }

    val outputShape = outputFormats()(0).shape

    val src = NativeData(inputShape, Memory.Format.any)
    val wei = NativeData(weightShape, Memory.Format.any)
    val bis = NativeData(Array(nOutputPlane), Memory.Format.x)
    val dst = NativeData(outputShape, Memory.Format.any)

    val desc = MklDnn.DilatedConvBackwardDataDescInit(
      AlgKind.ConvolutionDirect,
      src.getMemoryDescription(),
      wei.getMemoryDescription(), // TODO check correctness of strides and padding
      dst.getMemoryDescription(),
      Array(strideW, strideH), Array(dilationW_mkldnn, dilationH_mkldnn),
      paddingTL, paddingBR,
      MklDnn.PaddingKind.mkldnnPaddingZero)

    val backwardPrimDesc = MklDnn.PrimitiveDescCreate(desc, runtime.engine, forwardPrimDesc)

    val List(realDiffSrc, realWei, realDiffDst) =
      List(Query.DiffSrcPd, Query.WeightsPd, Query.DiffDstPd).map {x =>
        MemoryData.operationWant(backwardPrimDesc, x)
      }

    weightForBackwardMemoryData = realWei

    reorderManager.register(weight.heapData, realWei)

    // computing gradient input doesn't need the input
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

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    // if needed, reorder manager will reorder the wegiht to mkldnn wants
    weightForBackward = reorderManager.infer(Array(weight.heapData),
      Array(weightForBackwardMemoryData), weight.dense).asInstanceOf[DnnTensor[Float]]

    if (updateGradInputTensors == null) {
      val buffer = new ArrayBuffer[Tensor[Float]]()
      buffer.append(gradOutput.asInstanceOf[Tensor[Float]])
      buffer.append(weightForBackward)
      buffer.append(gradInput.asInstanceOf[Tensor[Float]])
      updateGradInputTensors = buffer.toArray
    }

    updateWithNewTensor(updateGradInputTensors, 0, gradOutput)
    updateWithNewTensor(updateGradInputTensors, 1, weightForBackward)

    MklDnnOps.streamSubmit(runtime.stream, 1, updateGradInputPrimitives,
      updateGradInputPrimitives.length, updateGradInputMemoryPrimitives, updateGradInputTensors)

    gradInput
  }

  override private[mkldnn] def initGradWPrimitives(grad: Array[MemoryData],
    phase: Phase): Array[MemoryData] = {
    val inputShape = inputFormats()(0).shape
    val outputShape = inputFormats()(0).shape

    val src = NativeData(inputShape, Memory.Format.any)
    val wei = NativeData(weightShape, Memory.Format.any)
    val bis = NativeData(Array(nOutputPlane), Memory.Format.x)

    val desc = MklDnn.DilatedConvBackwardWeightsDescInit(
      AlgKind.ConvolutionDirect,
      src.getMemoryDescription(),
      wei.getMemoryDescription(),
      bis.getMemoryDescription(),
      grad(0).getMemoryDescription(),
      Array(strideW, strideH), Array(dilationW_mkldnn, dilationH_mkldnn),
      paddingTL, paddingBR,
      MklDnn.PaddingKind.mkldnnPaddingZero)

    val gradWeightPrimDesc = MklDnn.PrimitiveDescCreate(desc, runtime.engine, forwardPrimDesc)

    // TODO here seems some errors ?????? check the realSrc format.
    val List(realSrc, realWei, realDiffDst) =
      List(Query.SrcPd, Query.DiffWeightsPd, Query.DiffDstPd).map { x =>
        MemoryData.operationWant(gradWeightPrimDesc, x)
      }

    // gradient weight should be the same format with weight
    val defaultWeightLayout = if (nGroup == 1) {
      Memory.Format.oihw
    } else {
      Memory.Format.goihw
    }

    gradWeight.setMemoryData(realWei,
      HeapData(gradWeight.dense.size(), defaultWeightLayout), runtime)
    gradBias.setMemoryData(bis,
      HeapData(gradBias.dense.size(), Memory.Format.x), runtime)

    // save the real input format accGradParameters wants, and register the reorder operation
    inputForAccMemoryData = realSrc
    reorderManager.register(inputFormats()(0), realSrc)

    val srcs = Array(realSrc.getPrimitive(runtime), realDiffDst.getPrimitive(runtime))
    val indexes = Array.fill(srcs.length)(0)
    val dsts = Array(realWei.getPrimitive(runtime), bis.getPrimitive(runtime))

    val primitive = MklDnn.PrimitiveCreate2(gradWeightPrimDesc, srcs, indexes, srcs.length,
      dsts, dsts.length)

    updateGradWMemoryPrimitives = srcs ++ dsts
    accGradientPrimitives = Array(primitive)

    _gradOutputFormatsForWeight = Array(realDiffDst)
    (_gradOutputFormatsForWeight)
  }

  override def accGradParameters(input: Activity, gradOutput: Activity): Unit = {
    // if needed, reorder manager will reorder input to mkldnn wants
    val inputTensor = if (input.isTensor) {
      input.toTensor[Float]
    } else {
      input.toTable.get[Tensor[Float]](_dim).get
    }
    inputForAcc = reorderManager.infer(Array(inputFormats()(0)),
      Array(inputForAccMemoryData), inputTensor).asInstanceOf[DnnTensor[Float]]

    if (updateGradWTensors == null) {
      val buffer = new ArrayBuffer[Tensor[Float]]()
      buffer.append(inputForAcc.asInstanceOf[Tensor[Float]])
      buffer.append(gradOutput.asInstanceOf[Tensor[Float]])
      buffer.append(gradWeight.native)
      buffer.append(gradBias.native)
      updateGradWTensors = buffer.toArray
    }

    updateWithNewTensor(updateGradWTensors, 0, inputForAcc)
    updateWithNewTensor(updateGradWTensors, 1, gradOutput)

    MklDnnOps.streamSubmit(runtime.stream, 1, accGradientPrimitives,
      accGradientPrimitives.length, updateGradWMemoryPrimitives, updateGradWTensors)

    gradWeight.sync()
    gradBias.sync()

    if (null != wRegularizer) {
      wRegularizer.accRegularization(weight.dense, gradWeight.dense, scaleW)
    }
    if (withBias && null != bRegularizer) {
      bRegularizer.accRegularization(bias.dense, gradBias.dense, scaleB)
    }
  }

  override def parameters(): (Array[Tensor[Float]], Array[Tensor[Float]]) = {
    if (withBias) {
      (Array(weight.dense, bias.dense), Array(gradWeight.dense, gradBias.dense))
    } else {
      (Array(weight.dense), Array(gradWeight.dense))
    }

  }

  // we need not implement it, because the grad parameters will clean by mkldnn
  override def zeroGradParameters(): Unit = {
  }

  override def release(): Unit = {
    super.release()
    List(weight, bias, gradWeight, gradBias).foreach(_.release())
    if (weightForBackward != null) { weightForBackward.release() }
  }

  override def setQuantize(value: Boolean): this.type = {
    needQuantize = value
    this
  }

  /**
   * TODO:
   *   (1) add calculation logic for Full padding
   *   (2) abstract and design an object type for return value, insteaof returnning
   *       the result as an int array
   * Calculate padding size
   * @param inputHeight height of input
   * @param inputWidth width of input
   * @param paddingType one of Same, Valid, Full
   * @return an int array of length 4 representing padding sizes
   *         (top, bottom, left, right)
   */
  private def getConvPaddingShape(inputHeight: Int, inputWidth: Int,
                                  paddingType: PaddingType.Value): ConvPaddingShape = {
    paddingType match {
      case PaddingType.Same =>
        getConvPaddingShape(inputHeight, inputWidth, kernelH, kernelW,
          strideH, strideW, dilationH, dilationW)
      case PaddingType.Custom => ConvPaddingShape(padH, padH, padW, padW)
      case _ => throw new IllegalArgumentException()
    }
  }

  /**
   * Helper function to get convolution padding shape for Same Padding
   * Steps:
   *   1). calculate the dimension of dilated kernel
   *       dilated kernel = kernel + (kernel - 1) * (dilation - 1)
   *   2). calculate the number of stride it would make
   *       number of stride = (input + stride - 1) / stride
   *   3). calculate the number of pad needed to make
   *       number of pad = start of last stride + dilated kernel - input
   *       start of last stride = (number of stride - 1) * stride
   *   4). assign the number of pad to left & right side
   * @param inputHeight height of input
   * @param inputWidth width of input
   * @param kernelHeight height of kernel
   * @param kernelWidth width of kernel
   * @param strideHeight height of stride
   * @param strideWidth width of stride
   * @param dilationHeight height of dilation
   * @param dilationWidth width of dilation
   * @return ConvPaddingShape
   */
  private def getConvPaddingShape(inputHeight: Int, inputWidth: Int,
                                  kernelHeight: Int, kernelWidth: Int,
                                  strideHeight: Int, strideWidth: Int,
                                  dilationHeight: Int, dilationWidth: Int
                                 ): ConvPaddingShape = {
    val dilatedKernelHeight = kernelHeight + (kernelHeight - 1) * (dilationHeight - 1)
    val dilatedKernelWidth = kernelWidth + (kernelWidth - 1) * (dilationWidth - 1)
    val numOfStrideHeight = (inputHeight + strideHeight - 1) / strideHeight
    val numOfStrideWidth = (inputWidth + strideWidth - 1) / strideWidth
    val padAlongHeight = Math.max(0,
      (numOfStrideHeight - 1) * strideHeight + dilatedKernelHeight - inputHeight)
    val padAlongWidth = Math.max(0,
      (numOfStrideWidth - 1) * strideWidth + dilatedKernelWidth - inputWidth)
    val (padTop, padBottom) = (padAlongHeight / 2, (padAlongHeight + 1) / 2)
    val (padLeft, padRight) = (padAlongWidth / 2, (padAlongWidth + 1) / 2)
    ConvPaddingShape(padTop, padBottom, padLeft, padRight)
  }



  /**
   * Calculate convolution output shape
   * Please try to keep the logic in consistent with MKL-DNN
   * Reffernce: https://github.com/intel/mkl-dnn/blob/master/src/common/convolution.cpp#L117
   * @param inputH height of input
   * @param inputW width of input
   * @return a ConvOutputShape object
   */
  private def getConvOutputShape(inputH: Int, inputW: Int,
                                 paddingShape: ConvPaddingShape): ConvOutputShape = {
    def getOutputLength(inputLength: Int, padLeft: Int, padRight: Int,
                        dilation: Int, kernelSize: Int, stride: Int): Int = {
      val kernelRange = 1 + (kernelSize - 1) * (dilation + 1)
      val outputLength = (inputLength - kernelRange + padLeft + padRight) / stride + 1
      return outputLength
    }
    val (padTop, padBottom, padLeft, padRight) = (
      paddingShape.top, paddingShape.bottom,
      paddingShape.left, paddingShape.right
    )
    val outputHeight = getOutputLength(inputH, padTop, padBottom,
      dilationH_mkldnn, kernelH, strideH)
    val outputWidth = getOutputLength(inputW, padLeft, padRight,
      dilationW_mkldnn, kernelW, strideH)

    ConvOutputShape(outputHeight, outputWidth)
  }


  /**
   * Get padding type
   * @return PaddingType
   */
  private def getPaddingType(): PaddingType.Value = {
    if (padH == -1 && padW == -1) {
      PaddingType.Same
    } else if (padH >= 0 && padW >= 0) {
      PaddingType.Custom
    } else {
      throw new IllegalArgumentException("Invalid padding")
    }
  }

}

object SpatialConvolution {
  def apply(
    nInputPlane: Int,
    nOutputPlane: Int,
    kW: Int,
    kH: Int,
    dW: Int = 1,
    dH: Int = 1,
    padW: Int = 0,
    padH: Int = 0,
    nGroup: Int = 1,
    propagateBack: Boolean = true,
    wRegularizer: Regularizer[Float] = null,
    bRegularizer: Regularizer[Float] = null,
    initWeight: Tensor[Float] = null,
    initBias: Tensor[Float] = null,
    initGradWeight: Tensor[Float] = null,
    initGradBias: Tensor[Float] = null,
    withBias: Boolean = true,
    format: DataFormat = DataFormat.NCHW,
    dilationW: Int = 1,
    dilationH: Int = 1): SpatialConvolution = {
    new SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW,
      dH, padW, padH, nGroup, propagateBack, wRegularizer, bRegularizer,
      initWeight, initBias, initGradWeight,
      initGradBias, withBias, format, dilationW, dilationH)
  }
}

object Scale {
  val S8_MAX = 127.0f
  val U8_MAX = 255.0f
}


/**
 * Enum for padding type
 * currently only support Same and Custom
 */
private[mkldnn] object PaddingType extends Enumeration {
  val Same, Custom = Value
}


/**
 * object to store meta for convolution padding shape
 * @param top
 * @param bottom
 * @param left
 * @param right
 */
private[mkldnn] case class ConvPaddingShape (
  top: Int,
  bottom: Int,
  left: Int,
  right: Int
)

/**
 * case class to store meta for convolution output shape
 * @param height
 * @param width
 */
private[mkldnn] case class ConvOutputShape (
  height: Int,
  width: Int
)
