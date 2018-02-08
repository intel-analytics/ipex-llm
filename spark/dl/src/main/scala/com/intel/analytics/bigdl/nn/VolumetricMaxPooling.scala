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
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import org.codehaus.jackson.map.DeserializationContext
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLModule}

import scala.reflect._
import scala.reflect.runtime.universe

/**
 * Applies 3D max-pooling operation in kTxkWxkH regions by step size dTxdWxdH.
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
  private var indices = Tensor[Float]()

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

    if (input.dim() == 4) {
      // non-batch mode
      output.resize(nslices, otime, oheight, owidth)
      indices.resize(nslices, otime, oheight, owidth)
      if (classTag[T] == classTag[Double]) {
        volumetricMaxPoolingForwardDouble(
          input.asInstanceOf[Tensor[Double]].storage().array(), input.storageOffset() - 1,
          output.asInstanceOf[Tensor[Double]].storage().array(), output.storageOffset() - 1,
          indices.storage().array(), indices.storageOffset() - 1,
          nslices, itime, iwidth, iheight, otime, owidth, oheight,
          kT, kW, kH, dT, dW, dH, padT, padW, padH)
      } else if (classTag[T] == classTag[Float]) {
        volumetricMaxPoolingForwardFloat(
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

      output.resize(nBatch, nslices, otime, oheight, owidth)
      indices.resize(nBatch, nslices, otime, oheight, owidth)

      var p = 0
      if (classTag[T] == classTag[Double]) {
        while (p < nBatch) {
          val curInput = input(p + 1)
          val curOutput = output(p + 1)
          val curIndices = indices(p + 1)
          volumetricMaxPoolingForwardDouble(
            curInput.asInstanceOf[Tensor[Double]].storage().array(),
            curInput.storageOffset() - 1,
            curOutput.asInstanceOf[Tensor[Double]].storage().array(),
            curOutput.storageOffset() - 1,
            curIndices.storage().array(),
            curIndices.storageOffset() - 1,
            nslices, itime, iwidth, iheight, otime, owidth, oheight,
            kT, kW, kH, dT, dW, dH, padT, padW, padH)
          p += 1
        }
      } else if (classTag[T] == classTag[Float]) {
        while (p < nBatch) {
          val curInput = input(p + 1)
          val curOutput = output(p + 1)
          val curIndices = indices(p + 1)
          volumetricMaxPoolingForwardFloat(
            curInput.asInstanceOf[Tensor[Float]].storage().array(),
            curInput.storageOffset() - 1,
            curOutput.asInstanceOf[Tensor[Float]].storage().array(),
            curOutput.storageOffset() - 1,
            curIndices.storage().array(),
            curIndices.storageOffset() - 1,
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
        volumetricMaxPoolingBackwardDouble(
          gradInput.asInstanceOf[Tensor[Double]].storage().array(), gradInput.storageOffset() - 1,
          gradOutput.asInstanceOf[Tensor[Double]].storage().array(), gradOutput.storageOffset() - 1,
          indices.storage().array(), indices.storageOffset() - 1,
          nslices, itime, iwidth, iheight, otime, owidth, oheight,
          dT, dW, dH, padT, padW, padH)
      } else if (classTag[T] == classTag[Float]) {
        volumetricMaxPoolingBackwardFloat(
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
      var p = 0

      if (classTag[T] == classTag[Double]) {
        while (p < nBatch) {
          val curGradInput = gradInput(p + 1)
          val curGradOutput = gradOutput(p + 1)
          val curIndices = indices(p + 1)
          volumetricMaxPoolingBackwardDouble(
            curGradInput.asInstanceOf[Tensor[Double]].storage().array(),
            curGradInput.storageOffset() - 1,
            curGradOutput.asInstanceOf[Tensor[Double]].storage().array(),
            curGradOutput.storageOffset() - 1,
            curIndices.storage().array(), curIndices.storageOffset() - 1,
            nslices, itime, iwidth, iheight, otime, owidth, oheight,
            dT, dW, dH, padT, padW, padH)
          p += 1
        }
      } else if (classTag[T] == classTag[Float]) {
        while (p < nBatch) {
          val curGradInput = gradInput(p + 1)
          val curGradOutput = gradOutput(p + 1)
          val curIndices = indices(p + 1)
          volumetricMaxPoolingBackwardFloat(
            curGradInput.asInstanceOf[Tensor[Float]].storage().array(),
            curGradInput.storageOffset() - 1,
            curGradOutput.asInstanceOf[Tensor[Float]].storage().array(),
            curGradOutput.storageOffset() - 1,
            curIndices.storage().array(), curIndices.storageOffset() - 1,
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

    if (!obj.isInstanceOf[VolumetricMaxPooling[T]]) {
      return false
    }
    val other = obj.asInstanceOf[VolumetricMaxPooling[T]]
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
      indices == other.indices
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
    hash = hash * seed + indices.hashCode()

    hash
  }

  override def toString(): String = {
    s"${getPrintName}($kT, $kW, $kH, $dT, $dW, $dH, $padT, $padW, $padH)"
  }

  override def clearState(): this.type = {
    super.clearState()
    indices.set()
    this
  }

  private def volumetricMaxPoolingForwardDouble(input: Array[Double], inputOffset: Int,
    output: Array[Double], outputOffset: Int,
    indices: Array[Float], indicesOffset: Int,
    nSlices: Int, iTime: Int, iWidth: Int, iHeight: Int, oTime: Int, oWidth: Int, oHeight: Int,
    kT: Int, kW: Int, kH: Int, dT: Int, dW: Int, dH: Int, padT: Int, padW: Int, padH: Int): Unit = {
    var k = 0
    while (k < nSlices) {
      var ti = 0
      while (ti < oTime) {
        var i = 0
        while (i < oHeight) {
          var j = 0
          while (j < oWidth) {
            var tstart = ti * dT - padT
            var hstart = i * dH - padH
            var wstart = j * dW - padW
            val kernelT = math.min(tstart + kT, kT)
            val kernelH = math.min(hstart + kH, kH)
            val kernelW = math.min(wstart + kW, kW)
            tstart = math.max(tstart, 0)
            hstart = math.max(hstart, 0)
            wstart = math.max(wstart, 0)

            val inputStart = inputOffset + k * iTime * iWidth * iHeight +
              tstart * iWidth * iHeight + hstart * iWidth + wstart

            var maxindex = 0 // default is 0
            var maxval = Double.MinValue
            var mx = 0
            var my = 0
            var mz = 0
            var z = 0
            while (z < kernelT) {
              var y = 0
              while (y < kernelH) {
                var x = 0
                while (x < kernelW) {
                  if ((tstart + z < iTime) && (hstart + y < iHeight) && (wstart + x < iWidth)) {
                    // k, z, y, x input indexers
                    val value = input(z * iWidth * iHeight + y * iWidth + x + inputStart)
                    if (value > maxval) {
                      maxval = value
                      // Store indices w.r.t the kernel dimension
                      mz = z + kT - kernelT
                      my = y + kH - kernelH
                      mx = x + kW - kernelW
                    }
                  }
                  x += 1
                }
                y += 1
              }
              z += 1
            }
            output(outputOffset + k * oTime * oWidth * oHeight
              + ti * oWidth * oHeight + i * oWidth + j) = maxval
            maxindex += ((mz & 0xff) << 24)
            maxindex += ((my & 0xff) << 16)
            maxindex += ((mx & 0xff) << 8)
            indices(indicesOffset + k * oTime * oWidth * oHeight
              + ti * oWidth * oHeight + i * oWidth + j) = maxindex
            j += 1
          }
          i += 1
        }
        ti += 1
      }
      k += 1
    }
  }

  private def volumetricMaxPoolingForwardFloat(input: Array[Float], inputOffset: Int,
    output: Array[Float], outputOffset: Int,
    indices: Array[Float], indicesOffset: Int,
    nSlices: Int, iTime: Int, iWidth: Int, iHeight: Int, oTime: Int, oWidth: Int, oHeight: Int,
    kT: Int, kW: Int, kH: Int, dT: Int, dW: Int, dH: Int, padT: Int, padW: Int, padH: Int): Unit = {
    var k = 0
    while (k < nSlices) {
      var ti = 0
      while (ti < oTime) {
        var i = 0
        while (i < oHeight) {
          var j = 0
          while (j < oWidth) {
            var tstart = ti * dT - padT
            var hstart = i * dH - padH
            var wstart = j * dW - padW
            val kernelT = math.min(tstart + kT, kT)
            val kernelH = math.min(hstart + kH, kH)
            val kernelW = math.min(wstart + kW, kW)
            tstart = math.max(tstart, 0)
            hstart = math.max(hstart, 0)
            wstart = math.max(wstart, 0)

            val inputStart = inputOffset + k * iTime * iWidth * iHeight +
              tstart * iWidth * iHeight + hstart * iWidth + wstart

            var maxindex = 0 // default is 0
            var maxval = Float.MinValue
            var mx = 0
            var my = 0
            var mz = 0
            var z = 0
            while (z < kernelT) {
              var y = 0
              while (y < kernelH) {
                var x = 0
                while (x < kernelW) {
                  if ((tstart + z < iTime) && (hstart + y < iHeight) && (wstart + x < iWidth)) {
                    // k, z, y, x input indexers
                    val value = input(z * iWidth * iHeight + y * iWidth + x + inputStart)
                    if (value > maxval) {
                      maxval = value
                      // Store indices w.r.t the kernel dimension
                      mz = z + kT - kernelT
                      my = y + kH - kernelH
                      mx = x + kW - kernelW
                    }
                  }
                  x += 1
                }
                y += 1
              }
              z += 1
            }
            output(outputOffset + k * oTime * oWidth * oHeight
              + ti * oWidth * oHeight + i * oWidth + j) = maxval
            maxindex += ((mz & 0xff) << 24)
            maxindex += ((my & 0xff) << 16)
            maxindex += ((mx & 0xff) << 8)
            indices(indicesOffset + k * oTime * oWidth * oHeight
              + ti * oWidth * oHeight + i * oWidth + j) = maxindex
            j += 1
          }
          i += 1
        }
        ti += 1
      }
      k += 1
    }
  }


  private def volumetricMaxPoolingBackwardDouble(gradInput: Array[Double], gradInputOffset: Int,
    gradOutput: Array[Double], gradOutputOffset: Int,
    indices: Array[Float], indicesOffset: Int,
    nslices: Int, itime: Int, iwidth: Int, iheight: Int,
    otime: Int, owidth: Int, oheight: Int,
    dT: Int, dW: Int, dH: Int, padT: Int, padW: Int, padH: Int): Unit = {
    var k = 0
    while (k < nslices) {
      val gradInputK = gradInputOffset + k * itime * iwidth * iheight
      val gradOutputK = gradOutputOffset + k * otime * owidth * oheight
      val indicesK = indicesOffset + k * otime * owidth * oheight
      var ti = 0
      while (ti < otime) {
        var i = 0
        while (i < oheight) {
          var j = 0
          while (j < owidth) {
            val maxIndex = indices(indicesK + ti * oheight * owidth + i * owidth + j).toInt
            val maxti = ((maxIndex >> 24) & 0xff) + ti * dT - padT
            val maxi = ((maxIndex >> 16) & 0xff) + i * dH - padH
            val maxj = ((maxIndex >> 8) & 0xff) + j * dW - padW
            gradInput(maxti * iheight * iwidth + maxi * iwidth + maxj + gradInputK) +=
              gradOutput(ti * oheight * owidth + i * owidth + j + gradOutputK)
            j += 1
          }
          i += 1
        }
        ti += 1
      }
      k += 1
    }
  }

  private def volumetricMaxPoolingBackwardFloat(gradInput: Array[Float], gradInputOffset: Int,
    gradOutput: Array[Float], gradOutputOffset: Int,
    indices: Array[Float], indicesOffset: Int,
    nslices: Int, itime: Int, iwidth: Int, iheight: Int,
    otime: Int, owidth: Int, oheight: Int,
    dT: Int, dW: Int, dH: Int, padT: Int, padW: Int, padH: Int): Unit = {
    var k = 0
    while (k < nslices) {
      val gradInputK = gradInputOffset + k * itime * iwidth * iheight
      val gradOutputK = gradOutputOffset + k * otime * owidth * oheight
      val indicesK = indicesOffset + k * otime * owidth * oheight
      var ti = 0
      while (ti < otime) {
        var i = 0
        while (i < oheight) {
          var j = 0
          while (j < owidth) {
            val maxIndex = indices(indicesK + ti * oheight * owidth + i * owidth + j).toInt
            val maxti = ((maxIndex >> 24) & 0xff) + ti * dT - padT
            val maxi = ((maxIndex >> 16) & 0xff) + i * dH - padH
            val maxj = ((maxIndex >> 8) & 0xff) + j * dW - padW
            gradInput(maxti * iheight * iwidth + maxi * iwidth + maxj + gradInputK) +=
              gradOutput(ti * oheight * owidth + i * owidth + j + gradOutputK)
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

object VolumetricMaxPooling extends ModuleSerializable {
  def apply[@specialized(Float, Double) T: ClassTag]
  (kT: Int, kW: Int, kH: Int, dT: Int, dW: Int, dH: Int,
    padT: Int = 0, padW: Int = 0, padH: Int = 0)(implicit ev: TensorNumeric[T])
  : VolumetricMaxPooling[T] = new VolumetricMaxPooling[T](kT, kW, kH, dT, dW, dH, padT, padW, padH)

  def apply[@specialized(Float, Double) T: ClassTag]
  (kT: Int, kW: Int, kH: Int)(implicit ev: TensorNumeric[T])
  : VolumetricMaxPooling[T] = new VolumetricMaxPooling[T](kT, kW, kH)

  override def doLoadModule[T: ClassTag](context : DeserializeContext)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {

    val maxPooling = super.doLoadModule(context).asInstanceOf[VolumetricMaxPooling[T]]
    val attrMap = context.bigdlModule.getAttrMap
    maxPooling.ceilMode = DataConverter.getAttributeValue(context,
      attrMap.get("ceilMode")).asInstanceOf[Boolean]
    maxPooling.indices = DataConverter.getAttributeValue(context,
      attrMap.get("indices")).asInstanceOf[Tensor[Float]]
    maxPooling
  }

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
                                            volumetricMaxBuilder : BigDLModule.Builder)
                                           (implicit ev: TensorNumeric[T]) : Unit = {
    val maxPooling = context.moduleData.module.asInstanceOf[VolumetricMaxPooling[T]]

    super.doSerializeModule(context, volumetricMaxBuilder)

    val ceilModeBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, ceilModeBuilder,
      maxPooling.ceilMode, universe.typeOf[Boolean])
    volumetricMaxBuilder.putAttr("ceilMode", ceilModeBuilder.build)

    val indicesBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context,
      indicesBuilder, maxPooling.indices, ModuleSerializer.tensorType)
    volumetricMaxBuilder.putAttr("indices", indicesBuilder.build)

  }
}
