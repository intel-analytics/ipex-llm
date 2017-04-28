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

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect._

/**
 * Applies 3D max-pooling operation in kTxkWxkH regions by step size dTxdWxdH steps.
 * The number of output features is equal to the number of input planes / dT.
 * The input can optionally be padded with zeros. Padding should be smaller than
 * half of kernel size. That is, padT < kT/2, padW < kW/2 and padH < kH/2
 * @param kT The kernel size
 * @param kW The kernel width
 * @param kH The kernel height
 * @param dT The step in the time dimension
 * @param dW The step in the width dimension
 * @param dH The step in the height dimension
 * @param padT The padding in the time dimension
 * @param padW The padding in the width dimension
 * @param padH The padding in the height dimension
 * @tparam T The numeric type in the criterion, usually which are [[Float]] or [[Double]]
 */
@SerialVersionUID(-4330398221120919890L)
class VolumetricMaxPooling[T: ClassTag](
  val kT: Int, val kW: Int, val kH: Int,
  val dT: Int, val dW: Int, val dH: Int,
  val padT: Int = 0, val padW: Int = 0, val padH: Int = 0)
  (implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  var ceilMode = false
  val indices = Tensor[Float]()

  def this(kT: Int, kW: Int, kH: Int)(implicit ev: TensorNumeric[T]) {
    this(kT, kW, kH, kT, kW, kH)
  }

  /**
   * set ceil mode
   * @return this
   */
  def ceil(): VolumetricMaxPooling[T] = {
    ceilMode = true
    this
  }

  /**
   * set floor mode
   * @return this
   */
  def floor(): VolumetricMaxPooling[T] = {
    ceilMode = false
    this
  }


  require(kT > 0 && kW > 0 && kH > 0,
    s"kernel size should be greater than zero, but got kT: $kT kH: $kH kW: $kW")

  require(dT > 0 && dW > 0 && dH > 0,
    s"stride should be greater than zero, but got dT: $dT dH: $dH dW: $dW")

  require(kT / 2 >= padT && kW / 2 >= padW && kH / 2 >= padH,
    "pad should be smaller than half of kernel size, but got " +
      s"kT: $kT kH: $kH kW: $kW, padT: $padT, padW: $padW, padH: $padH")

  /**
   * Computes the output using the current parameter set of the class and input. This function
   * returns the result which is stored in the output field.
   * @param input
   * @return
   */
  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 4 || input.dim() == 5,
      s"4D or 5D (batch mode) tensor expected for input, but got: ${ input.dim() }")
    require(input.isContiguous(), "input is not contiguous")
    val dimt = input.dim() - 2
    val dimh = input.dim() - 1
    val dimw = input.dim()
    val nslices = input.size(input.dim() - 3)
    val itime = input.size(dimt)
    val iheight = input.size(dimh)
    val iwidth = input.size(dimw)


    var otime: Int = 0
    var oheight: Int = 0
    var owidth: Int = 0
    if (ceilMode) {
      otime = math.ceil(1.0 * (itime - kT + 2 * padT) / dT).toInt + 1
      oheight = math.ceil(1.0 * (iheight - kH + 2 * padH) / dH).toInt + 1
      owidth = math.ceil(1.0 * (iwidth - kW + 2 * padW) / dW).toInt + 1
    }
    else {
      otime = math.floor(1.0 * (itime - kT + 2 * padT) / dT).toInt + 1
      oheight = math.floor(1.0 * (iheight - kH + 2 * padH) / dH).toInt + 1
      owidth = math.floor(1.0 * (iwidth - kW + 2 * padW) / dW).toInt + 1
    }
    if (padT != 0 || padW != 0 || padH != 0) {
      // ensure that the last pooling starts inside the image
      if ((otime - 1) * dT >= itime + padT) otime -= 1
      if ((oheight - 1) * dH >= iheight + padH) oheight -= 1
      if ((owidth - 1) * dW >= iwidth + padW) owidth -= 1
    }
    require(otime >= 1 && owidth >= 1 && oheight >= 1,
      s"Given input size: (${ nslices }x${ itime }x${ iheight }x${ iwidth })." +
        s" Calculated output size:" +
        s" (${ nslices }x${ otime }x${ oheight }x${ owidth }). Output size is too small")

    // non-batch mode
    if (input.dim() == 4) {
      output.resize(nslices, otime, oheight, owidth)
      indices.resize(nslices, otime, oheight, owidth)
      if (classTag[T] == classTag[Double]) {
        NNPrimitive.volumetricMaxPoolingForwardDouble(
          input.asInstanceOf[Tensor[Double]].storage().array(), input.storageOffset() - 1,
          output.asInstanceOf[Tensor[Double]].storage().array(), output.storageOffset() - 1,
          indices.storage().array(), indices.storageOffset() - 1,
          nslices, itime, iwidth, iheight, otime, owidth, oheight,
          kT, kW, kH, dT, dW, dH, padT, padW, padH)
      } else if (classTag[T] == classTag[Float]) {
        NNPrimitive.volumetricMaxPoolingForwardFloat(
          input.asInstanceOf[Tensor[Float]].storage().array(), input.storageOffset() - 1,
          output.asInstanceOf[Tensor[Float]].storage().array(), output.storageOffset() - 1,
          indices.storage().array(), indices.storageOffset() - 1,
          nslices, itime, iwidth, iheight, otime, owidth, oheight,
          kT, kW, kH, dT, dW, dH, padT, padW, padH)
      } else {
        throw new IllegalArgumentException("currently only support type float or double")
      }
    } else {
      // batch mode
      val nBatch = input.size(1)

      val istride = nslices * itime * iwidth * iheight
      val ostride = nslices * otime * owidth * oheight

      output.resize(nBatch, nslices, otime, oheight, owidth)
      indices.resize(nBatch, nslices, otime, oheight, owidth)

      var p = 0
      if (classTag[T] == classTag[Double]) {
        while (p < nBatch) {
          NNPrimitive.volumetricMaxPoolingForwardDouble(
            input.asInstanceOf[Tensor[Double]].storage().array(),
            input.storageOffset() - 1 + p * istride,
            output.asInstanceOf[Tensor[Double]].storage().array(),
            output.storageOffset() - 1 + p * ostride,
            indices.storage().array(),
            indices.storageOffset() - 1 + p * ostride,
            nslices, itime, iwidth, iheight, otime, owidth, oheight,
            kT, kW, kH, dT, dW, dH, padT, padW, padH)
          p += 1
        }
      } else if (classTag[T] == classTag[Float]) {
        while (p < nBatch) {
          NNPrimitive.volumetricMaxPoolingForwardFloat(
            input.asInstanceOf[Tensor[Float]].storage().array(),
            input.storageOffset() - 1 + p * istride,
            output.asInstanceOf[Tensor[Float]].storage().array(),
            output.storageOffset() - 1 + p * ostride,
            indices.storage().array(),
            indices.storageOffset() - 1 + p * ostride,
            nslices, itime, iwidth, iheight, otime, owidth, oheight,
            kT, kW, kH, dT, dW, dH, padT, padW, padH)
          p += 1
        }
      } else {
        throw new IllegalArgumentException("currently only support type float or double")
      }

    }
    output
  }

  /**
   * Computing the gradient of the module with respect to its own input. This is returned in
   * gradInput. Also, the gradInput state variable is updated accordingly.
   * @param input
   * @param gradOutput
   * @return
   */
  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val dimn = input.dim() - 3
    val dimt = input.dim() - 2
    val dimh = input.dim() - 1
    val dimw = input.dim()

    val nslices = input.size(dimn)
    val itime = input.size(dimt)
    val iheight = input.size(dimh)
    val iwidth = input.size(dimw)
    val otime = gradOutput.size(dimt)
    val oheight = gradOutput.size(dimh)
    val owidth = gradOutput.size(dimw)

    gradInput.resizeAs(input).zero()
    require(gradOutput.isContiguous())

    if (input.dim() == 4) {
      if (classTag[T] == classTag[Double]) {
        NNPrimitive.volumetricMaxPoolingBackwardDouble(
          gradInput.asInstanceOf[Tensor[Double]].storage().array(), gradInput.storageOffset() - 1,
          gradOutput.asInstanceOf[Tensor[Double]].storage().array(), gradOutput.storageOffset() - 1,
          indices.storage().array(), indices.storageOffset() - 1,
          nslices, itime, iwidth, iheight, otime, owidth, oheight,
          dT, dW, dH, padT, padW, padH)
      } else if (classTag[T] == classTag[Float]) {
        NNPrimitive.volumetricMaxPoolingBackwardFloat(
          gradInput.asInstanceOf[Tensor[Float]].storage().array(), gradInput.storageOffset() - 1,
          gradOutput.asInstanceOf[Tensor[Float]].storage().array(), gradOutput.storageOffset() - 1,
          indices.storage().array(), indices.storageOffset() - 1,
          nslices, itime, iwidth, iheight, otime, owidth, oheight,
          dT, dW, dH, padT, padW, padH)
      } else {
        throw new IllegalArgumentException("currently only support type float or double")
      }
    }
    else {
      // batch mode
      val nBatch = input.size(1)

      val istride = nslices * itime * iwidth * iheight
      val ostride = nslices * otime * owidth * oheight
      var p = 0

      if (classTag[T] == classTag[Double]) {
        while (p < nBatch) {
          val curGradInput = gradInput(p + 1)
          val curGradOutput = gradOutput(p + 1)
          val curIndices = indices(p + 1)
          NNPrimitive.volumetricMaxPoolingBackwardDouble(
            curGradInput.asInstanceOf[Tensor[Double]].storage().array(),
            curGradInput.storageOffset() - 1,
            curGradOutput.asInstanceOf[Tensor[Double]].storage().array(),
            curGradOutput.storageOffset() - 1,
            curIndices.storage().array(), curIndices.storageOffset() - 1,
            nslices, itime, iwidth, iheight, otime, owidth, oheight,
            dT, dW, dH, padT, padW, padH)

          require(curIndices.storageOffset() - 1 == indices.storageOffset() - 1 + p * ostride)
          p += 1
        }
      } else if (classTag[T] == classTag[Float]) {
        while (p < nBatch) {
          NNPrimitive.volumetricMaxPoolingBackwardFloat(
            gradInput.asInstanceOf[Tensor[Float]].storage().array(),
            gradInput.storageOffset() - 1 + p * istride,
            gradOutput.asInstanceOf[Tensor[Float]].storage().array(),
            gradOutput.storageOffset() - 1 + p * ostride,
            indices.storage().array(), indices.storageOffset() - 1 + p * ostride,
            nslices, itime, iwidth, iheight, otime, owidth, oheight,
            dT, dW, dH, padT, padW, padH)
          p += 1
        }
      } else {
        throw new IllegalArgumentException("currently only support type float or double")
      }
    }
    gradInput
  }

  override def clearState(): this.type = {
    super.clearState()
    indices.set()
    this
  }
}

object VolumetricMaxPooling {
  def apply[@specialized(Float, Double) T: ClassTag]
  (kT: Int, kW: Int, kH: Int, dT: Int, dW: Int, dH: Int,
    padT: Int = 0, padW: Int = 0, padH: Int = 0)(implicit ev: TensorNumeric[T])
  : VolumetricMaxPooling[T] = new VolumetricMaxPooling[T](kT, kW, kH, dT, dW, dH, padT, padW)

  def apply[@specialized(Float, Double) T: ClassTag]
  (kT: Int, kW: Int, kH: Int)(implicit ev: TensorNumeric[T])
  : VolumetricMaxPooling[T] = new VolumetricMaxPooling[T](kT, kW, kH)
}
