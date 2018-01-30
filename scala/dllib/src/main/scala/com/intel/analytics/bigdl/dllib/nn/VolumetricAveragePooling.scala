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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer._
import org.codehaus.jackson.map.DeserializationContext
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLModule}

import scala.reflect._
import scala.reflect.runtime.universe

/**
 * Applies 3D average-pooling operation in kTxkWxkH regions by step size dTxdWxdH.
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
 * @param countIncludePad Whether to include padding when dividing the
 *                        number of elements in pooling region
 * @param ceilMode Whether the output size is to be ceiled or floored
 * @tparam T The numeric type in the criterion, usually which are [[Float]] or [[Double]]
 */
@SerialVersionUID(- 7829953407414301872L)
class VolumetricAveragePooling[T: ClassTag](
  val kT: Int, val kW: Int, val kH: Int,
  val dT: Int, val dW: Int, val dH: Int,
  val padT: Int = 0, val padW: Int = 0, val padH: Int = 0,
  private var countIncludePad: Boolean = true,
  private var ceilMode: Boolean = false)
  (implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  def this(kT: Int, kW: Int, kH: Int)(implicit ev: TensorNumeric[T]) {
    this(kT, kW, kH, kT, kW, kH)
  }

  /**
   * set ceil mode
   * @return this
   */
  def ceil(): VolumetricAveragePooling[T] = {
    ceilMode = true
    this
  }

  /**
   * set floor mode
   * @return this
   */
  def floor(): VolumetricAveragePooling[T] = {
    ceilMode = false
    this
  }

  /**
   * set countIncludePad to true
   * @return this
   */
  def setCountIncludePad(): VolumetricAveragePooling[T] = {
    countIncludePad = true
    this
  }

  /**
   * set countIncludePad to false
   * @return this
   */
  def setCountExcludePad(): VolumetricAveragePooling[T] = {
    countIncludePad = false
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
    require(input.size(dimw) >= kW && input.size(dimh) >= kH && input.size(dimt) >= kT,
      s"input image (T: ${input.size(dimt)} H: ${input.size(dimh)} W: ${input.size(dimw)}) " +
        s"smaller than kernel size (kT: $kT kH: $kH kW: $kW)")

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

    if (input.dim() == 4) {
      // non-batch mode
      output.resize(nslices, otime, oheight, owidth)
      if (classTag[T] == classTag[Double]) {
        volumetricAveragePoolingForwardDouble(
          input.asInstanceOf[Tensor[Double]].storage().array(), input.storageOffset() - 1,
          output.asInstanceOf[Tensor[Double]].storage().array(), output.storageOffset() - 1,
          countIncludePad, nslices, itime, iwidth, iheight, otime, owidth, oheight,
          kT, kW, kH, dT, dW, dH, padT, padW, padH)
      } else if (classTag[T] == classTag[Float]) {
        volumetricAveragePoolingForwardFloat(
          input.asInstanceOf[Tensor[Float]].storage().array(), input.storageOffset() - 1,
          output.asInstanceOf[Tensor[Float]].storage().array(), output.storageOffset() - 1,
          countIncludePad, nslices, itime, iwidth, iheight, otime, owidth, oheight,
          kT, kW, kH, dT, dW, dH, padT, padW, padH)
      } else {
        throw new IllegalArgumentException("currently only support type float or double")
      }
    } else {
      // batch mode
      val nBatch = input.size(1)

      output.resize(nBatch, nslices, otime, oheight, owidth)

      var p = 0
      if (classTag[T] == classTag[Double]) {
        while (p < nBatch) {
          val curInput = input(p + 1)
          val curOutput = output(p + 1)
          volumetricAveragePoolingForwardDouble(
            curInput.asInstanceOf[Tensor[Double]].storage().array(),
            curInput.storageOffset() - 1,
            curOutput.asInstanceOf[Tensor[Double]].storage().array(),
            curOutput.storageOffset() - 1, countIncludePad,
            nslices, itime, iwidth, iheight, otime, owidth, oheight,
            kT, kW, kH, dT, dW, dH, padT, padW, padH)
          p += 1
        }
      } else if (classTag[T] == classTag[Float]) {
        while (p < nBatch) {
          val curInput = input(p + 1)
          val curOutput = output(p + 1)
          volumetricAveragePoolingForwardFloat(
            curInput.asInstanceOf[Tensor[Float]].storage().array(),
            curInput.storageOffset() - 1,
            curOutput.asInstanceOf[Tensor[Float]].storage().array(),
            curOutput.storageOffset() - 1, countIncludePad,
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
    require(gradOutput.isContiguous(), "gradOutput is not contiguous")

    if (input.dim() == 4) {
      // non-batch mode
      if (classTag[T] == classTag[Double]) {
        volumetricAveragePoolingBackwardDouble(
          gradInput.asInstanceOf[Tensor[Double]].storage().array(), gradInput.storageOffset() - 1,
          gradOutput.asInstanceOf[Tensor[Double]].storage().array(), gradOutput.storageOffset() - 1,
          countIncludePad, nslices, itime, iwidth, iheight, otime, owidth, oheight,
          dT, dW, dH, padT, padW, padH)
      } else if (classTag[T] == classTag[Float]) {
        volumetricAveragePoolingBackwardFloat(
          gradInput.asInstanceOf[Tensor[Float]].storage().array(), gradInput.storageOffset() - 1,
          gradOutput.asInstanceOf[Tensor[Float]].storage().array(), gradOutput.storageOffset() - 1,
          countIncludePad, nslices, itime, iwidth, iheight, otime, owidth, oheight,
          dT, dW, dH, padT, padW, padH)
      } else {
        throw new IllegalArgumentException("currently only support type float or double")
      }
    }
    else {
      // batch mode
      val nBatch = input.size(1)
      var p = 0

      if (classTag[T] == classTag[Double]) {
        while (p < nBatch) {
          val curGradInput = gradInput(p + 1)
          val curGradOutput = gradOutput(p + 1)
          volumetricAveragePoolingBackwardDouble(
            curGradInput.asInstanceOf[Tensor[Double]].storage().array(),
            curGradInput.storageOffset() - 1,
            curGradOutput.asInstanceOf[Tensor[Double]].storage().array(),
            curGradOutput.storageOffset() - 1, countIncludePad,
            nslices, itime, iwidth, iheight, otime, owidth, oheight,
            dT, dW, dH, padT, padW, padH)
          p += 1
        }
      } else if (classTag[T] == classTag[Float]) {
        while (p < nBatch) {
          val curGradInput = gradInput(p + 1)
          val curGradOutput = gradOutput(p + 1)
          volumetricAveragePoolingBackwardFloat(
            curGradInput.asInstanceOf[Tensor[Float]].storage().array(),
            curGradInput.storageOffset() - 1,
            curGradOutput.asInstanceOf[Tensor[Float]].storage().array(),
            curGradOutput.storageOffset() - 1, countIncludePad,
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

  override def equals(obj: Any): Boolean = {

    if (!super.equals(obj)) {
      return false
    }

    if (!obj.isInstanceOf[VolumetricAveragePooling[T]]) {
      return false
    }
    val other = obj.asInstanceOf[VolumetricAveragePooling[T]]
    if (this.eq(other)) {
      return true
    }

    kT == other.kT &&
      kW == other.kW &&
      kH == other.kH &&
      dT == other.dT &&
      dW == other.dW &&
      dH == other.dH &&
      padT == other.padT &&
      padW == other.padW &&
      padH == other.padH &&
      ceilMode == other.ceilMode &&
      countIncludePad == other.countIncludePad
  }

  override def hashCode(): Int = {
    val seed = 37
    var hash = super.hashCode()
    hash = hash * seed + kT.hashCode()
    hash = hash * seed + kW.hashCode()
    hash = hash * seed + kH.hashCode()
    hash = hash * seed + dT.hashCode()
    hash = hash * seed + dW.hashCode()
    hash = hash * seed + dH.hashCode()
    hash = hash * seed + padT.hashCode()
    hash = hash * seed + padW.hashCode()
    hash = hash * seed + padH.hashCode()
    hash = hash * seed + ceilMode.hashCode()
    hash = hash * seed + countIncludePad.hashCode()

    hash
  }

  override def toString(): String = {
    s"${getPrintName}($kT, $kW, $kH, $dT, $dW, $dH, $padT, $padW, $padH, " +
      s"$countIncludePad, $ceilMode)"
  }

  override def clearState(): this.type = {
    super.clearState()
    this
  }

  private def volumetricAveragePoolingForwardDouble(input: Array[Double], inputOffset: Int,
    output: Array[Double], outputOffset: Int, countIncludePad: Boolean,
    nSlices: Int, iTime: Int, iWidth: Int, iHeight: Int, oTime: Int, oWidth: Int, oHeight: Int,
    kT: Int, kW: Int, kH: Int, dT: Int, dW: Int, dH: Int, padT: Int, padW: Int, padH: Int): Unit = {
    var k = 0
    while (k < nSlices) {
      val inputStart = inputOffset + k * iTime * iWidth * iHeight
      val outputStart = outputOffset + k * oTime * oWidth * oHeight
      var ti = 0
      while (ti < oTime) {
        var i = 0
        while (i < oHeight) {
          var j = 0
          while (j < oWidth) {
            var tstart = ti * dT - padT
            var hstart = i * dH - padH
            var wstart = j * dW - padW
            var tend = math.min(tstart + kT, iTime + padT)
            var hend = math.min(hstart + kH, iHeight + padH)
            var wend = math.min(wstart + kW, iWidth + padW)
            var poolSize = (tend - tstart) * (hend - hstart) * (wend - wstart)
            tstart = math.max(tstart, 0)
            hstart = math.max(hstart, 0)
            wstart = math.max(wstart, 0)
            tend = math.min(tend, iTime)
            hend = math.min(hend, iHeight)
            wend = math.min(wend, iWidth)

            val divide_factor = if (countIncludePad) poolSize
            else (tend - tstart) * (hend - hstart) * (wend - wstart)

            var sum = 0.0
            var z = tstart
            while (z < tend) {
              var y = hstart
              while (y < hend) {
                var x = wstart
                while (x < wend) {
                  val value = input(z * iWidth * iHeight + y * iWidth + x + inputStart)
                  sum += value
                  x += 1
                }
                y += 1
              }
              z += 1
            }
            output(ti * oWidth * oHeight + i * oWidth + j + outputStart) = sum / divide_factor
            j += 1
          }
          i += 1
        }
        ti += 1
      }
      k += 1
    }
  }

  private def volumetricAveragePoolingForwardFloat(input: Array[Float], inputOffset: Int,
    output: Array[Float], outputOffset: Int, countIncludePad: Boolean,
    nSlices: Int, iTime: Int, iWidth: Int, iHeight: Int, oTime: Int, oWidth: Int, oHeight: Int,
    kT: Int, kW: Int, kH: Int, dT: Int, dW: Int, dH: Int, padT: Int, padW: Int, padH: Int): Unit = {
    var k = 0
    while (k < nSlices) {
      val inputStart = inputOffset + k * iTime * iWidth * iHeight
      val outputStart = outputOffset + k * oTime * oWidth * oHeight
      var ti = 0
      while (ti < oTime) {
        var i = 0
        while (i < oHeight) {
          var j = 0
          while (j < oWidth) {
            var tstart = ti * dT - padT
            var hstart = i * dH - padH
            var wstart = j * dW - padW
            var tend = math.min(tstart + kT, iTime + padT)
            var hend = math.min(hstart + kH, iHeight + padH)
            var wend = math.min(wstart + kW, iWidth + padW)
            var poolSize = (tend - tstart) * (hend - hstart) * (wend - wstart)
            tstart = math.max(tstart, 0)
            hstart = math.max(hstart, 0)
            wstart = math.max(wstart, 0)
            tend = math.min(tend, iTime)
            hend = math.min(hend, iHeight)
            wend = math.min(wend, iWidth)

            val divide_factor = if (countIncludePad) poolSize
                                else (tend - tstart) * (hend - hstart) * (wend - wstart)

            var sum = 0.0f
            var z = tstart
            while (z < tend) {
              var y = hstart
              while (y < hend) {
                var x = wstart
                while (x < wend) {
                  val value = input(z * iWidth * iHeight + y * iWidth + x + inputStart)
                  sum += value
                  x += 1
                }
                y += 1
              }
              z += 1
            }
            output(ti * oWidth * oHeight + i * oWidth + j + outputStart) = sum / divide_factor
            j += 1
          }
          i += 1
        }
        ti += 1
      }
      k += 1
    }
  }


  private def volumetricAveragePoolingBackwardDouble(gradInput: Array[Double], gradInputOffset: Int,
   gradOutput: Array[Double], gradOutputOffset: Int, countIncludePad: Boolean,
   nslices: Int, iTime: Int, iWidth: Int, iHeight: Int,
   oTime: Int, oWidth: Int, oHeight: Int,
   dT: Int, dW: Int, dH: Int, padT: Int, padW: Int, padH: Int): Unit = {
    var k = 0
    while (k < nslices) {
      val gradInputK = gradInputOffset + k * iTime * iWidth * iHeight
      val gradOutputK = gradOutputOffset + k * oTime * oWidth * oHeight
      var ti = 0
      while (ti < oTime) {
        var i = 0
        while (i < oHeight) {
          var j = 0
          while (j < oWidth) {
            var tstart = ti * dT - padT
            var hstart = i * dH - padH
            var wstart = j * dW - padW
            var tend = math.min(tstart + kT, iTime + padT)
            var hend = math.min(hstart + kH, iHeight + padH)
            var wend = math.min(wstart + kW, iWidth + padW)
            val poolSize = (tend - tstart) * (hend - hstart) * (wend - wstart)
            tstart = math.max(tstart, 0)
            hstart = math.max(hstart, 0)
            wstart = math.max(wstart, 0)
            tend = math.min(tend, iTime)
            hend = math.min(hend, iHeight)
            wend = math.min(wend, iWidth)

            val divide_factor = if (countIncludePad) poolSize
                                else (tend - tstart) * (hend - hstart) * (wend - wstart)

            val s = gradOutput(ti * oWidth * oHeight + i * oWidth + j + gradOutputK)
            var z = tstart
            while (z < tend) {
              var y = hstart
              while (y < hend) {
                var x = wstart
                while (x < wend) {
                  gradInput(z * iWidth * iHeight + y * iWidth +
                    x + gradInputK) += s / divide_factor
                  x += 1
                }
                y += 1
              }
              z += 1
            }
            j += 1
          }
          i += 1
        }
        ti += 1
      }
      k += 1
    }
  }

  private def volumetricAveragePoolingBackwardFloat(gradInput: Array[Float], gradInputOffset: Int,
    gradOutput: Array[Float], gradOutputOffset: Int, countIncludePad: Boolean,
    nslices: Int, iTime: Int, iWidth: Int, iHeight: Int,
    oTime: Int, oWidth: Int, oHeight: Int,
    dT: Int, dW: Int, dH: Int, padT: Int, padW: Int, padH: Int): Unit = {
    var k = 0
    while (k < nslices) {
      val gradInputK = gradInputOffset + k * iTime * iWidth * iHeight
      val gradOutputK = gradOutputOffset + k * oTime * oWidth * oHeight
      var ti = 0
      while (ti < oTime) {
        var i = 0
        while (i < oHeight) {
          var j = 0
          while (j < oWidth) {
            var tstart = ti * dT - padT
            var hstart = i * dH - padH
            var wstart = j * dW - padW
            var tend = math.min(tstart + kT, iTime + padT)
            var hend = math.min(hstart + kH, iHeight + padH)
            var wend = math.min(wstart + kW, iWidth + padW)
            val poolSize = (tend - tstart) * (hend - hstart) * (wend - wstart)
            tstart = math.max(tstart, 0)
            hstart = math.max(hstart, 0)
            wstart = math.max(wstart, 0)
            tend = math.min(tend, iTime)
            hend = math.min(hend, iHeight)
            wend = math.min(wend, iWidth)

            val divide_factor = if (countIncludePad) poolSize
                                else (tend - tstart) * (hend - hstart) * (wend - wstart)

            val s = gradOutput(ti * oWidth * oHeight + i * oWidth + j + gradOutputK)
            var z = tstart
            while (z < tend) {
              var y = hstart
              while (y < hend) {
                var x = wstart
                while (x < wend) {
                  gradInput(z * iWidth * iHeight + y * iWidth +
                    x + gradInputK) += s / divide_factor
                  x += 1
                }
                y += 1
              }
              z += 1
            }
            j += 1
          }
          i += 1
        }
        ti += 1
      }
      k += 1
    }
  }
}

object VolumetricAveragePooling extends ModuleSerializable {
  def apply[@specialized(Float, Double) T: ClassTag]
  (kT: Int, kW: Int, kH: Int, dT: Int, dW: Int, dH: Int,
   padT: Int = 0, padW: Int = 0, padH: Int = 0,
   countIncludePad: Boolean = true, ceilMode: Boolean = false)(implicit ev: TensorNumeric[T])
  : VolumetricAveragePooling[T] =
    new VolumetricAveragePooling[T](kT, kW, kH, dT, dW, dH,
      padT, padW, padH, countIncludePad, ceilMode)

}
