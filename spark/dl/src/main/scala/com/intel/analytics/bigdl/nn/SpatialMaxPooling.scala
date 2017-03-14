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
import com.intel.analytics.bigdl.utils.Engine

import scala.reflect._

@SerialVersionUID(2277597677473874749L)
class SpatialMaxPooling[T: ClassTag](
  val kW: Int, val kH: Int, val dW: Int, val dH: Int, val padW: Int = 0, val padH: Int = 0)
  (implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  var ceil_mode = false
  val indices = Tensor[T]()

  def this(kW: Int, kH: Int)(implicit ev: TensorNumeric[T]) {
    this(kW, kH, kW, kH)
  }

  def ceil(): SpatialMaxPooling[T] = {
    ceil_mode = true
    this
  }

  def floor(): SpatialMaxPooling[T] = {
    ceil_mode = false
    this
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 3 || input.dim() == 4,
      "SpatialMaxPooling: " + ErrorInfo.constrainInputAs3DOrBatch)
    val dimw = input.dim()
    val dimh = input.dim() - 1
    require(input.size(dimw) >= kW - padW && input.size(dimh) >= kH - padH,
      "input smaller than kernel size")
    require(kW / 2 >= padW && kH / 2 >= padH, "pad should be smaller than half of kernel size")
    val nslices = input.size(dimh - 1)
    val iheight = input.size(dimh)
    val iwidth = input.size(dimw)
    var oheight: Int = 0
    var owidth: Int = 0
    if (ceil_mode) {
      oheight = math.ceil(1.0 * (iheight - kH + 2 * padH) / dH).toInt + 1
      owidth = math.ceil(1.0 * (iwidth - kW + 2 * padW) / dW).toInt + 1
    }
    else {
      oheight = math.floor(1.0 * (iheight - kH + 2 * padH) / dH).toInt + 1
      owidth = math.floor(1.0 * (iwidth - kW + 2 * padW) / dW).toInt + 1
    }

    if (padW != 0 || padH != 0) {
      if ((oheight - 1) * dH >= iheight + padH) oheight -= 1
      if ((owidth - 1) * dW >= iwidth + padW) owidth -= 1
    }

    if (input.dim() == 3) {
      output.resize(Array(nslices, oheight, owidth))
      /* indices will contain the locations for each output point */
      indices.resize(Array(nslices, oheight, owidth))
      if (classTag[T] == classTag[Double]) {
        NNPrimitive.maxPoolingForwardDouble(
          input.asInstanceOf[Tensor[Double]].storage().array(), input.storageOffset() - 1,
          output.asInstanceOf[Tensor[Double]].storage().array(), output.storageOffset() - 1,
          indices.asInstanceOf[Tensor[Double]].storage().array(), indices.storageOffset() - 1,
          nslices, iwidth, iheight, owidth, oheight, kW, kH, dW, dH, padW, padH)
      } else if (classTag[T] == classTag[Float]) {
        NNPrimitive.maxPoolingForwardFloat(
          input.asInstanceOf[Tensor[Float]].storage().array(), input.storageOffset() - 1,
          output.asInstanceOf[Tensor[Float]].storage().array(), output.storageOffset() - 1,
          indices.asInstanceOf[Tensor[Float]].storage().array(), indices.storageOffset() - 1,
          nslices, iwidth, iheight, owidth, oheight, kW, kH, dW, dH, padW, padH)
      } else {
        throw new IllegalArgumentException
      }
    }
    else {
      val nbatch = input.size(1)
      output.resize(Array(nbatch, nslices, oheight, owidth))
      indices.resize(Array(nbatch, nslices, oheight, owidth))
      if (classTag[T] == classTag[Double]) {
        Engine.model.invokeAndWait(
          (1 to nbatch).map(i => () => {
            val curInput = input(i)
            val curOutput = output(i)
            val curIndices = indices(i)
            NNPrimitive.maxPoolingForwardDouble(
              curInput.asInstanceOf[Tensor[Double]].storage().array(),
              curInput.storageOffset() - 1,
              curOutput.asInstanceOf[Tensor[Double]].storage().array(),
              curOutput.storageOffset() - 1,
              curIndices.asInstanceOf[Tensor[Double]].storage().array(),
              curIndices.storageOffset() - 1,
              nslices, iwidth, iheight, owidth, oheight,
              kW, kH, dW, dH, padW, padH
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
              curInput.asInstanceOf[Tensor[Float]].storage().array(),
              curInput.storageOffset() - 1,
              curOutput.asInstanceOf[Tensor[Float]].storage().array(),
              curOutput.storageOffset() - 1,
              curIndices.asInstanceOf[Tensor[Float]].storage().array(),
              curIndices.storageOffset() - 1,
              nslices, iwidth, iheight, owidth, oheight,
              kW, kH, dW, dH, padW, padH
            )
          })
        )
      } else {
        throw new IllegalArgumentException
      }
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val dimw = input.dim()
    val dimh = input.dim() - 1
    require(input.size(dimw) >= kW - padW && input.size(dimh) >= kH - padH,
      "input smaller than kernel size")
    require(kW / 2 >= padW && kH / 2 >= padH, "pad should be smaller than half of kernel size")
    val nslices = input.size(dimh - 1)
    val iheight = input.size(dimh)
    val iwidth = input.size(dimw)
    val oheight: Int = gradOutput.size(dimh)
    val owidth: Int = gradOutput.size(dimw)
    gradInput.resizeAs(input)
    gradInput.zero()
    if (input.dim() == 3) {
      if (classTag[T] == classTag[Double]) {
        NNPrimitive.maxPoolingBackwardDouble(
          gradInput.asInstanceOf[Tensor[Double]].storage().array(), gradInput.storageOffset() - 1,
          gradOutput.asInstanceOf[Tensor[Double]].storage().array(), gradOutput.storageOffset() - 1,
          indices.asInstanceOf[Tensor[Double]].storage().array(), indices.storageOffset() - 1,
          nslices, iwidth, iheight, owidth, oheight)
      } else if (classTag[T] == classTag[Float]) {
        NNPrimitive.maxPoolingBackwardFloat(
          gradInput.asInstanceOf[Tensor[Float]].storage().array(), gradInput.storageOffset() - 1,
          gradOutput.asInstanceOf[Tensor[Float]].storage().array(), gradOutput.storageOffset() - 1,
          indices.asInstanceOf[Tensor[Float]].storage().array(), indices.storageOffset() - 1,
          nslices, iwidth, iheight, owidth, oheight)
      } else {
        throw new IllegalArgumentException
      }
    }
    else {
      val nbacth = input.size(1)
      if (classTag[T] == classTag[Double]) {
        Engine.model.invokeAndWait(
          (1 to nbacth).map(k => () => {
            val curGradInput = gradInput(k)
            val curGradOutput = gradOutput(k)
            val curIndices = indices(k)
            NNPrimitive.maxPoolingBackwardDouble(
              curGradInput.asInstanceOf[Tensor[Double]].storage().array(),
              curGradInput.storageOffset() - 1,
              curGradOutput.asInstanceOf[Tensor[Double]].storage().array(),
              curGradOutput.storageOffset() - 1,
              curIndices.asInstanceOf[Tensor[Double]].storage().array(),
              curIndices.storageOffset() - 1,
              nslices, iwidth, iheight, owidth, oheight
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
              curGradInput.asInstanceOf[Tensor[Float]].storage().array(),
              curGradInput.storageOffset() - 1,
              curGradOutput.asInstanceOf[Tensor[Float]].storage().array(),
              curGradOutput.storageOffset() - 1,
              curIndices.asInstanceOf[Tensor[Float]].storage().array(),
              curIndices.storageOffset() - 1,
              nslices, iwidth, iheight, owidth, oheight
            )
          })
        )
      } else {
        throw new IllegalArgumentException
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
      ceil_mode == other.ceil_mode &&
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
    hash = hash * seed + ceil_mode.hashCode()
    hash = hash * seed + indices.hashCode()

    hash
  }

  override def toString(): String = {
    s"nn.SpatialMaxPooling($kW, $kH, $dW, $dH, $padW, $padH)"
  }

  override def clearState(): this.type = {
    super.clearState()
    indices.set()
    this
  }
}

object SpatialMaxPooling {
  def apply[@specialized(Float, Double) T: ClassTag](
      kW: Int,
      kH: Int,
      dW: Int,
      dH: Int,
      padW: Int = 0,
      padH: Int = 0)(implicit ev: TensorNumeric[T]): SpatialMaxPooling[T] = {
    new SpatialMaxPooling[T](kW, kH, dW, dH, padW, padH)
  }
}
