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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, DataFormat, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.utils.serializer._
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLModule}

import scala.reflect._
import scala.reflect.runtime.universe

/**
 * Applies 2D max-pooling operation in kWxkH regions by step size dWxdH steps.
 * The number of output features is equal to the number of input planes.
 * If the input image is a 3D tensor nInputPlane x height x width,
 * the output image size will be nOutputPlane x oheight x owidth where
 * owidth  = op((width  + 2*padW - kW) / dW + 1)
 * oheight = op((height + 2*padH - kH) / dH + 1)
 * op is a rounding operator. By default, it is floor.
 * It can be changed by calling :ceil() or :floor() methods.
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
 * @param kW              kernel width
 * @param kH              kernel height
 * @param dW              step size in width
 * @param dH              step size in height
 * @param padW            padding in width
 * @param padH            padding in height
 * @param format          DataFormat.NCHW or DataFormat.NHWC, indicating the input
 *                        data format
 */
@SerialVersionUID(2277597677473874749L)
class SpatialMaxPooling[T: ClassTag](
  val kW: Int, val kH: Int, val dW: Int, val dH: Int, val padW: Int = 0, val padH: Int = 0,
  val format: DataFormat = DataFormat.NCHW)(implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  var ceilMode = false
  val indices = Tensor[T]()

  def this(kW: Int, kH: Int)(implicit ev: TensorNumeric[T]) {
    this(kW, kH, kW, kH, format = DataFormat.NCHW)
  }

  /**
   * set ceil mode
   * @return this
   */
  def ceil(): SpatialMaxPooling[T] = {
    ceilMode = true
    this
  }

  /**
   * set floor mode
   * @return this
   */
  def floor(): SpatialMaxPooling[T] = {
    ceilMode = false
    this
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 3 || input.dim() == 4,
      "SpatialMaxPooling: " + ErrorInfo.constrainInputAs3DOrBatch)

    val (dimh, dimw, dimc) = format.getHWCDims(input.dim())

    val nInputPlane = input.size(dimc)
    val inputHeight = input.size(dimh)
    val inputWidth = input.size(dimw)

    val sizes =
      if (padW == -1 && padH == -1) {
        // no ceil/floor mode in SAME padding
        Utils.getSAMEOutSizeAndPadding(inputHeight, inputWidth, dH, dW, kH, kW)
      } else {
        require(inputWidth >= kW - padW && inputHeight >= kH - padH,
          "input smaller than kernel size" +
            s"input size(${input.size(dimw)},${input.size(dimh)})" +
            s"kernel size(${kW-padW},${kH-padH})")
        require(kW / 2 >= padW && kH / 2 >= padH, "pad should be smaller than half of kernel size" +
          s"pad size($padW,$padH)" +
          s"kernel size($kW, $kH)")
        Utils.getOutSizeAndPadding(inputHeight, inputWidth, dH, dW, kH, kW, padH, padW, ceilMode)
      }

    val padTop = sizes(0)
    val padBottom = sizes(1)
    val padLeft = sizes(2)
    val padRight = sizes(3)
    val oHeight = sizes(4)
    val oWidth = sizes(5)

    if (ceilMode && padW == 0 && (inputWidth - kW) % dW == 0) {
      ceilMode = false // The ceil mode is not needed.
    }

    if (input.dim() == 3) {
      format match {
        case DataFormat.NCHW =>
          output.resize(Array(nInputPlane, oHeight, oWidth))
          /* indices will contain the locations for each output point */
          indices.resize(Array(nInputPlane, oHeight, oWidth))
          if (classTag[T] == classTag[Double]) {
            NNPrimitive.maxPoolingForwardDouble(
              input.asInstanceOf[Tensor[Double]],
              output.asInstanceOf[Tensor[Double]],
              indices.asInstanceOf[Tensor[Double]],
              oWidth, oHeight, kW, kH, dW, dH, padLeft, padTop)
          } else if (classTag[T] == classTag[Float]) {
            NNPrimitive.maxPoolingForwardFloat(
              input.asInstanceOf[Tensor[Float]],
              output.asInstanceOf[Tensor[Float]],
              indices.asInstanceOf[Tensor[Float]],
              oWidth, oHeight, kW, kH, dW, dH, padLeft, padTop)
          } else {
            throw new IllegalArgumentException
          }
        case DataFormat.NHWC =>
          output.resize(Array(oHeight, oWidth, nInputPlane))
          /* indices will contain the locations for each output point */
          indices.resize(Array(oHeight, oWidth, nInputPlane))
          if (classTag[T] == classTag[Double]) {
            NNPrimitive.maxPoolingForwardDoubleNHWC(
              input.asInstanceOf[Tensor[Double]],
              output.asInstanceOf[Tensor[Double]],
              indices.asInstanceOf[Tensor[Double]],
              oWidth, oHeight, kW, kH, dW, dH, padLeft, padTop)
          } else if (classTag[T] == classTag[Float]) {
            NNPrimitive.maxPoolingForwardFloatNHWC(
              input.asInstanceOf[Tensor[Float]],
              output.asInstanceOf[Tensor[Float]],
              indices.asInstanceOf[Tensor[Float]],
              oWidth, oHeight, kW, kH, dW, dH, padLeft, padTop)
          } else {
            throw new IllegalArgumentException
          }
      }
    } else {
      val nbatch = input.size(1)
      format match {
        case DataFormat.NCHW =>
          output.resize(Array(nbatch, nInputPlane, oHeight, oWidth))
          indices.resize(Array(nbatch, nInputPlane, oHeight, oWidth))
          if (classTag[T] == classTag[Double]) {
            Engine.model.invokeAndWait(
              (1 to nbatch).map(i => () => {
                val curInput = input(i)
                val curOutput = output(i)
                val curIndices = indices(i)
                NNPrimitive.maxPoolingForwardDouble(
                  curInput.asInstanceOf[Tensor[Double]],
                  curOutput.asInstanceOf[Tensor[Double]],
                  curIndices.asInstanceOf[Tensor[Double]],
                  oWidth, oHeight,
                  kW, kH, dW, dH, padLeft, padTop
                )
              })
            )
          } else if (classTag[T] == classTag[Float]) {
            Engine.model.invokeAndWait(
              (1 to nbatch).map(i => () => {
                val curInput = input(i)
                val curOutput = output(i)
                val curIndices = indices(i)
                NNPrimitive.maxPoolingForwardFloat(
                  curInput.asInstanceOf[Tensor[Float]],
                  curOutput.asInstanceOf[Tensor[Float]],
                  curIndices.asInstanceOf[Tensor[Float]],
                  oWidth, oHeight,
                  kW, kH, dW, dH, padLeft, padTop
                )
              })
            )
          } else {
            throw new IllegalArgumentException
          }
        case DataFormat.NHWC =>
          output.resize(Array(nbatch, oHeight, oWidth, nInputPlane))
          indices.resize(Array(nbatch, oHeight, oWidth, nInputPlane))
          if (classTag[T] == classTag[Double]) {
            Engine.model.invokeAndWait(
              (1 to nbatch).map(i => () => {
                val curInput = input(i)
                val curOutput = output(i)
                val curIndices = indices(i)
                NNPrimitive.maxPoolingForwardDoubleNHWC(
                  curInput.asInstanceOf[Tensor[Double]],
                  curOutput.asInstanceOf[Tensor[Double]],
                  curIndices.asInstanceOf[Tensor[Double]],
                  oWidth, oHeight,
                  kW, kH, dW, dH, padLeft, padTop
                )
              })
            )
          } else if (classTag[T] == classTag[Float]) {
            Engine.model.invokeAndWait(
              (1 to nbatch).map(i => () => {
                val curInput = input(i)
                val curOutput = output(i)
                val curIndices = indices(i)
                NNPrimitive.maxPoolingForwardFloatNHWC(
                  curInput.asInstanceOf[Tensor[Float]],
                  curOutput.asInstanceOf[Tensor[Float]],
                  curIndices.asInstanceOf[Tensor[Float]],
                  oWidth, oHeight,
                  kW, kH, dW, dH, padLeft, padTop
                )
              })
            )
          } else {
            throw new IllegalArgumentException
          }
      }
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {

    val (dimh, dimw, dimc) = format.getHWCDims(input.dim())

    val oHeight: Int = gradOutput.size(dimh)
    val oWidth: Int = gradOutput.size(dimw)
    gradInput.resizeAs(input)
    gradInput.zero()
    if (input.dim() == 3) {
      format match {
        case DataFormat.NCHW =>
          if (classTag[T] == classTag[Double]) {
            NNPrimitive.maxPoolingBackwardDouble(
              gradInput.asInstanceOf[Tensor[Double]],
              gradOutput.asInstanceOf[Tensor[Double]],
              indices.asInstanceOf[Tensor[Double]],
              oWidth, oHeight)
          } else if (classTag[T] == classTag[Float]) {
            NNPrimitive.maxPoolingBackwardFloat(
              gradInput.asInstanceOf[Tensor[Float]],
              gradOutput.asInstanceOf[Tensor[Float]],
              indices.asInstanceOf[Tensor[Float]],
              oWidth, oHeight)
          } else {
            throw new IllegalArgumentException
          }
        case DataFormat.NHWC =>
          if (classTag[T] == classTag[Double]) {
            NNPrimitive.maxPoolingBackwardDoubleNHWC(
              gradInput.asInstanceOf[Tensor[Double]],
              gradOutput.asInstanceOf[Tensor[Double]],
              indices.asInstanceOf[Tensor[Double]],
              oWidth, oHeight)
          } else if (classTag[T] == classTag[Float]) {
            NNPrimitive.maxPoolingBackwardFloatNHWC(
              gradInput.asInstanceOf[Tensor[Float]],
              gradOutput.asInstanceOf[Tensor[Float]],
              indices.asInstanceOf[Tensor[Float]],
              oWidth, oHeight)
          } else {
            throw new IllegalArgumentException
          }
      }
    }
    else {
      val nbacth = input.size(1)
      format match {
        case DataFormat.NCHW =>
          if (classTag[T] == classTag[Double]) {
            Engine.model.invokeAndWait(
              (1 to nbacth).map(k => () => {
                val curGradInput = gradInput(k)
                val curGradOutput = gradOutput(k)
                val curIndices = indices(k)
                NNPrimitive.maxPoolingBackwardDouble(
                  curGradInput.asInstanceOf[Tensor[Double]],
                  curGradOutput.asInstanceOf[Tensor[Double]],
                  curIndices.asInstanceOf[Tensor[Double]],
                  oWidth, oHeight
                )
              })
            )
          } else if (classTag[T] == classTag[Float]) {
            Engine.model.invokeAndWait(
              (1 to nbacth).map(k => () => {
                val curGradInput = gradInput(k)
                val curGradOutput = gradOutput(k)
                val curIndices = indices(k)
                NNPrimitive.maxPoolingBackwardFloat(
                  curGradInput.asInstanceOf[Tensor[Float]],
                  curGradOutput.asInstanceOf[Tensor[Float]],
                  curIndices.asInstanceOf[Tensor[Float]],
                  oWidth, oHeight
                )
              })
            )
          } else {
            throw new IllegalArgumentException
          }
        case DataFormat.NHWC =>
          if (classTag[T] == classTag[Double]) {
            Engine.model.invokeAndWait(
              (1 to nbacth).map(k => () => {
                val curGradInput = gradInput(k)
                val curGradOutput = gradOutput(k)
                val curIndices = indices(k)
                NNPrimitive.maxPoolingBackwardDoubleNHWC(
                  curGradInput.asInstanceOf[Tensor[Double]],
                  curGradOutput.asInstanceOf[Tensor[Double]],
                  curIndices.asInstanceOf[Tensor[Double]],
                  oWidth, oHeight
                )
              })
            )
          } else if (classTag[T] == classTag[Float]) {
            Engine.model.invokeAndWait(
              (1 to nbacth).map(k => () => {
                val curGradInput = gradInput(k)
                val curGradOutput = gradOutput(k)
                val curIndices = indices(k)
                NNPrimitive.maxPoolingBackwardFloatNHWC(
                  curGradInput.asInstanceOf[Tensor[Float]],
                  curGradOutput.asInstanceOf[Tensor[Float]],
                  curIndices.asInstanceOf[Tensor[Float]],
                  oWidth, oHeight
                )
              })
            )
          } else {
            throw new IllegalArgumentException
          }
      }

    }
    gradInput
  }

  override def equals(obj: Any): Boolean = {

    if (!super.equals(obj)) {
      return false
    }

    if (!obj.isInstanceOf[SpatialMaxPooling[T]]) {
      return false
    }
    val other = obj.asInstanceOf[SpatialMaxPooling[T]]
    if (this.eq(other)) {
      return true
    }

    kW == other.kW &&
      kH == other.kH &&
      dW == other.dW &&
      dH == other.dH &&
      padW == other.padW &&
      padH == other.padH &&
      ceilMode == other.ceilMode &&
      indices == other.indices
  }

  override def hashCode() : Int = {
    val seed = 37
    var hash = super.hashCode()
    hash = hash * seed + kW.hashCode()
    hash = hash * seed + kH.hashCode()
    hash = hash * seed + dW.hashCode()
    hash = hash * seed + dH.hashCode()
    hash = hash * seed + padW.hashCode()
    hash = hash * seed + padH.hashCode()
    hash = hash * seed + ceilMode.hashCode()
    hash = hash * seed + indices.hashCode()

    hash
  }

  override def toString(): String = {
    s"${getPrintName}($kW, $kH, $dW, $dH, $padW, $padH)"
  }

  override def clearState(): this.type = {
    super.clearState()
    indices.set()
    this
  }
}

object SpatialMaxPooling extends ModuleSerializable {
  def apply[@specialized(Float, Double) T: ClassTag](
      kW: Int,
      kH: Int,
      dW: Int = 1,
      dH: Int = 1,
      padW: Int = 0,
      padH: Int = 0,
      format: DataFormat = DataFormat.NCHW)
      (implicit ev: TensorNumeric[T]): SpatialMaxPooling[T] = {
    new SpatialMaxPooling[T](kW, kH, dW, dH, padW, padH, format)
  }

  override def doLoadModule[T: ClassTag](context: DeserializeContext)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {
    val maxPooling = super.doLoadModule(context)
    val attrMap = context.bigdlModule.getAttrMap
    val ceil_mode = DataConverter.
      getAttributeValue(context, attrMap.get("ceil_mode")).
      asInstanceOf[Boolean]
    if (ceil_mode) {
      maxPooling.asInstanceOf[SpatialMaxPooling[T]].ceil()
    }
    maxPooling
  }

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
                                              maxPoolingBuilder : BigDLModule.Builder)
                                           (implicit ev: TensorNumeric[T]) : Unit = {

    super.doSerializeModule(context, maxPoolingBuilder)
    val maxPooling = context.moduleData.module.asInstanceOf[SpatialMaxPooling[T]]
    val ceilBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, ceilBuilder,
      maxPooling.ceilMode, universe.typeOf[Boolean])
    maxPoolingBuilder.putAttr("ceil_mode", ceilBuilder.build)

  }
}
