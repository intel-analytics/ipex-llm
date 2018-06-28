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

package com.intel.analytics.bigdl.nn.quantized

import com.intel.analytics.bigdl.bigquant.BigQuant
import com.intel.analytics.bigdl.nn
import com.intel.analytics.bigdl.nn.ErrorInfo
import com.intel.analytics.bigdl.nn.abstractnn.{DataFormat, Initializable}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import com.intel.analytics.bigdl.utils.serializer.{DeserializeContext, ModuleData, SerializeContext}
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.runtime.universe
import scala.reflect.ClassTag
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLModule}

@SerialVersionUID(- 8008252944905538960L)
private[bigdl] class SpatialConvolution[T: ClassTag](
  val nInputPlane: Int, // The number of expected input planes in the image given into forward()
  val nOutputPlane: Int, // The number of output planes the convolution layer will produce.
  val kernelW: Int, // The kernel width of the convolution
  val kernelH: Int, // The kernel height of the convolution
  val strideW: Int = 1, // The step of the convolution in the width dimension.
  val strideH: Int = 1, // The step of the convolution in the height dimension
  val padW: Int = 0, // The additional zeros added per width to the input planes.
  val padH: Int = 0, // The additional zeros added per height to the input planes.
  val nGroup: Int = 1, // Kernel group number
  val format: DataFormat = DataFormat.NCHW
)(implicit ev: TensorNumeric[T]) extends QuantizedModule[T](nOutputPlane) with Initializable {

  require(nInputPlane % nGroup == 0, "Number of input channels should be multiples of group.")
  require(nOutputPlane % nGroup == 0, "Number of output channels should be multiples of group.")

  private val data: QuantizedTensor[T] = QuantizedDummyTensor[T]()
  val bias: Tensor[T] = Tensor[T](nOutputPlane)

  val quantFormat: Int = if (format == DataFormat.NCHW) {
    BigQuant.NCHW
  } else {
    BigQuant.NHWC
  }

  val params = ConvWeightParams(nOutputPlane / nGroup, nInputPlane / nGroup, kernelH, kernelW,
    quantFormat)

  val weight: Array[Tensor[T]] = {
    val array = new Array[Tensor[T]](nGroup)
    for (i <- 0 until nGroup) {
      array(i) = QuantizedTensor[T](Tensor[T](Array(nGroup, kernelH, kernelW, nInputPlane / nGroup,
        nOutputPlane / nGroup)), params)
    }
    array
  }

  val dilationHeight = 1
  val dilationWidth = 1

  protected def initWeightAndBias(weightFP32: Tensor[T], biasFP32: Tensor[T]): this.type = {
    if (biasFP32 != null) {
      bias.copy(biasFP32)
    } else {
      bias.fill(ev.fromType(0)) // TODO bias may be null, at that time, we should not initialize it
    }

    // dilated convolution has no group option
    val weightTmp = if (format == DataFormat.NHWC) {
      val groupWeight = weightFP32.view(Array(nGroup, kernelH, kernelW, nInputPlane / nGroup,
        nOutputPlane / nGroup))

      nn.Utils.shuffle(groupWeight, Array(1, 5, 2, 3, 4))
    } else {
      weightFP32.view(Array(nGroup, nOutputPlane / nGroup, nInputPlane / nGroup,
        kernelH, kernelW))
    }

    for (i <- 1 to nGroup) {
      val groupWeight = weightTmp.select(1, i)
      ev.getType() match {
        case FloatType =>
          weight(i - 1).asInstanceOf[QuantizedTensor[T]].release()
          weight(i - 1) = QuantizedTensor[T](groupWeight, params)
        case _ => throw new UnsupportedOperationException(s"Only support Float for quantized model")
      }
    }

    this
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 3 || input.dim() == 4,
      "quantized.SpatialConvolution: " + ErrorInfo.constrainInputAs3DOrBatch)
    require(input.isContiguous())

    val (dimHeight, dimWidth, channelDim) = format.getHWCDims(input.dim())
    require(input.size(channelDim) == nInputPlane, s"input channel size " +
      s"${input.size(channelDim)} is not the same as nInputPlane $nInputPlane")

    val inputWidth = input.size(dimWidth)
    val inputHeight = input.size(dimHeight)

    val sizes =
      if (padW == -1 && padH == -1) {
        nn.Utils.getSAMEOutSizeAndPadding(inputHeight, inputWidth, strideH, strideW, kernelH,
          kernelW)
      } else {
        nn.Utils.getOutSizeAndPadding(inputHeight, inputWidth, strideH, strideW,
          kernelH, kernelW, padH, padW, ceilMode = false, dilationWidth = dilationWidth,
          dilationHeight = dilationHeight)
      }

    val padTop = sizes(0)
    val padBottom = sizes(1)
    val padLeft = sizes(2)
    val padRight = sizes(3)
    val outputHeight = sizes(4)
    val outputWidth = sizes(5)

    val batchSize = if (input.dim() == 3) {
      output.resize(nn.Utils.getOutputShape(outputHeight, outputWidth, nOutputPlane,
        format = format))
      1 // 3D input, batchSize set to 1
    } else {
      val batch = input.size(1)
      output.resize(nn.Utils.getOutputShape(outputHeight, outputWidth, nOutputPlane, batch, format))
      batch
    }

    val params = ConvDataParams(nInputPlane / nGroup, kernelH, kernelW,
        strideH, strideW, padTop, padLeft, dilationHeight, dilationWidth, 1,
        inputHeight, inputWidth)

    if (data.params == null || data.params != params) {
      data.release()
      data.set(QuantizedTensor[T](input.size(), params))
    }

    ev.getType() match {
      case FloatType =>
        var batch = 0
        while (batch < batchSize) {
            im2ColAndGemmFloat(batch)
          batch += 1
        }
      case _ => throw new UnsupportedOperationException(s"Only support Float for quantized model")
    }

    @inline def im2ColAndGemmFloat(batch: Int): Unit = {
      val batchOutput = output.select(1, batch + 1)
      val batchInput = input.select(1, batch + 1)
      val channel = if (input.dim() == 3) { channelDim } else { channelDim - 1 }

      var group = 0
      while (group < nGroup) {
        val groupBatchOutput = batchOutput.narrow(channel, group * nOutputPlane / nGroup + 1,
          nOutputPlane / nGroup)
        val groupBatchInput = batchInput.narrow(channel, group * nInputPlane / nGroup + 1,
          nInputPlane / nGroup)
        val groupWeight = weight(group).asInstanceOf[QuantizedTensor[T]]
        val offset = 0

        groupIm2ColGemm(groupBatchInput, groupBatchOutput, groupWeight, offset)

        group += 1
      }
    }

    @inline def groupIm2ColGemm(input: Tensor[T], output: Tensor[T],
      weight: QuantizedTensor[T], offset: Int): Unit = {
      val inputArray = input.storage().array().asInstanceOf[Array[Float]]
      val inputOffset = input.storageOffset() - 1

      val outputArray = output.storage().array().asInstanceOf[Array[Float]]
      val outputOffset = output.storageOffset() - 1

      val biasArray = bias.storage().array().asInstanceOf[Array[Float]]
      val biasOffset = bias.storageOffset() - 1 + offset

      val weightSumArray = weight.sumOfRow.asInstanceOf[Array[Float]]
      val weightSumOffset = offset

      BigQuant.ConvDataInit(
        data.getNativeStorage, inputArray, inputOffset,
        nInputPlane / nGroup, kernelH, kernelW, strideH, strideW, padTop, padLeft,
        dilationHeight, dilationWidth, 1, inputHeight, inputWidth, QuantParams.THRESHOLD,
        quantFormat)

      BigQuant.MixPrecisionGEMM(
        quantFormat, weight.getNativeStorage, data.getNativeStorage,
        outputArray, outputOffset, weightSumArray, weightSumOffset,
        biasArray, biasOffset, 1, nOutputPlane / nGroup, outputHeight, outputWidth,
        QuantParams.FAULT_TOLERANCE)
    }

    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Doesn't updateGradInput for quantized model")
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (weight :+ bias, Array.fill[Tensor[T]](nGroup + 1)(empty)) // nGroup's weight + bias
  }

  override def equals(obj: Any): Boolean = {
    if (!super.equals(obj)) {
      return false
    }
    if (!obj.isInstanceOf[SpatialConvolution[T]]) {
      return false
    }
    val other = obj.asInstanceOf[SpatialConvolution[T]]
    if (this.eq(other)) {
      return true
    }

    nInputPlane == other.nInputPlane &&
      nOutputPlane == other.nOutputPlane &&
      kernelW == other.kernelW &&
      kernelH == other.kernelH &&
      strideW == other.strideW &&
      strideH == other.strideH &&
      padW == other.padW &&
      padH == other.padH &&
      nGroup == other.nGroup &&
      weight == other.weight &&
      bias == other.bias
  }

  override def hashCode(): Int = {
    val seed = 37
    var hash = super.hashCode()
    hash = hash * seed + nInputPlane.hashCode()
    hash = hash * seed + nOutputPlane.hashCode()
    hash = hash * seed + kernelW.hashCode()
    hash = hash * seed + kernelH.hashCode()
    hash = hash * seed + strideW.hashCode()
    hash = hash * seed + strideH.hashCode()
    hash = hash * seed + padW.hashCode()
    hash = hash * seed + padH.hashCode()
    hash = hash * seed + bias.hashCode()
    hash = hash * seed + weight.hashCode()

    hash
  }

  override def clearState() : this.type = {
    super.clearState()
    this
  }

  override def toString(): String = {
    s"quantized.SpatialConvolution($nInputPlane -> $nOutputPlane, $kernelW x" +
      s" $kernelH, $strideW, $strideH, $padW, $padH, $nGroup)"
  }

  override def release(): Unit = {
    weight.foreach(_.asInstanceOf[QuantizedTensor[T]].release())
    data.release()
  }
}

object SpatialConvolution extends QuantSerializer {
  def apply[@specialized(Float) T: ClassTag](
    nInputPlane: Int,
    nOutputPlane: Int,
    kernelW: Int,
    kernelH: Int,
    strideW: Int = 1,
    strideH: Int = 1,
    padW: Int = 0,
    padH: Int = 0,
    nGroup: Int = 1,
    initWeight: Tensor[T] = null,
    initBias: Tensor[T] = null,
    format: DataFormat = DataFormat.NCHW
  )(implicit ev: TensorNumeric[T]): SpatialConvolution[T] = {
    val conv = new SpatialConvolution[T](nInputPlane, nOutputPlane, kernelW, kernelH,
      strideW, strideH, padW, padH, nGroup, format)
    conv.initWeightAndBias(initWeight, initBias)
  }

  override def serializeWeight[T: ClassTag](context: SerializeContext[T],
    modelBuilder: BigDLModule.Builder)(implicit ev: TensorNumeric[T]): Unit = {
    val module = context.moduleData.module
    val conv = module.asInstanceOf[SpatialConvolution[T]]
    val weightBuilder = AttrValue.newBuilder
    ev.getType() match {
      case FloatType =>
        DataConverter.setAttributeValue(context, weightBuilder, conv.weight,
          universe.typeOf[Array[Tensor[Float]]])
      case _ => throw new UnsupportedOperationException(s"Only support Float for quantized model")
    }
    modelBuilder.putAttr("weights", weightBuilder.build)
  }

  override def loadWeight[T: ClassTag](context: DeserializeContext,
    moduleData: ModuleData[T])(implicit ev: TensorNumeric[T]): Unit = {
    val conv = moduleData.module.asInstanceOf[SpatialConvolution[T]]
    val attrMap = context.bigdlModule.getAttrMap
    val weights = DataConverter.getAttributeValue(context, attrMap.get("weights"))
      .asInstanceOf[Array[Tensor[T]]]
    for (i <- 0 until conv.weight.length) {
      conv.weight(i).asInstanceOf[QuantizedTensor[T]].release()
      conv.weight(i).set(weights(i))
    }
  }
}

