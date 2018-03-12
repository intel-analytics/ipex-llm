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
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

/**
 * Upsampling layer for 3D inputs.
 * Repeats the 1st, 2nd and 3rd dimensions
 * of the data by size[0], size[1] and size[2] respectively.
 * The input data is assumed to be of the form `minibatch x channels x depth x height x width`.
 *
 * @param size Repeats the depth, height, width dimensions of the data by
 *             size[0], size[1] and size[2] respectively.
 * @tparam T The numeric type in the criterion, usually which are [[Float]] or [[Double]]
 */
@SerialVersionUID(3462228835945094156L)
class UpSampling3D[T: ClassTag](val size: Array[Int])
  (implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  require(size != null && size.length == 3, "the size should be 3 dims")

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    require(input.length == 5,
      s"UpSampling3D requires 5D input, but got input dim ${input.length}")
    Shape(input(0), input(1), input(2)*size(0), input(3)*size(1), input(4)*size(2))
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 5, "only supports 5d tensors")
    require(input.isContiguous(), "input need to be contiguous")

    val inputDepth = input.size(3)
    val inputHeight = input.size(4)
    val inputWidth = input.size(5)

    val dT = size(0)
    val dH = size(1)
    val dW = size(2)

    val outputDepth = inputDepth * dT
    val outputHeight = inputHeight * dH
    val outputWidth = inputWidth * dW

    output.resize(input.size(1), input.size(2), outputDepth, outputHeight, outputWidth)

    // dims
    val idim = input.dim()
    val xDim = idim - 1
    val yDim = idim - 2
    val zDim = idim - 3

    val osz0 = output.size(1)
    val osz1 = output.size(2)
    val osz2 = output.size(3)
    val osz3 = output.size(4)
    val osz4 = output.size(5)

    // get strides
    val is = input.stride()
    val os = output.stride()

    // get raw pointers
    val pin = input.storage().array()
    val inOffset = input.storageOffset() - 1
    val pout = output.storage().array()
    val outOffset = output.storageOffset() - 1

    // perform the upsampling
    var i0, i1, i2, i3, i4, isrc, idst = 0
    val iout = new Array[Int](5) // Output indices
    val iin = new Array[Int](5) // Input indices

    i0 = 0
    while (i0 < osz0) {
      iout(0) = i0
      iin(0) = i0
      i1 = 0
      while (i1 < osz1) {
        iout(1) = i1
        iin(1) = i1
        i2 = 0
        while (i2 < osz2) {
          iout(2) = i2
          iin(2) = i2
          i3 = 0
          while (i3 < osz3) {
            iout(3) = i3
            iin(3) = i3
            i4 = 0
            while (i4 < osz4) {
              iout(4) = i4
              iin(4) = i4
              // set the indices for the upsampled dimensions
              iin(xDim) = iout(xDim) / dW
              iin(yDim) = iout(yDim) / dH
              iin(zDim) = iout(zDim) / dT

              idst = i0 * os(0) + i1 * os(1) + i2 * os(2) + i3 * os(3)
              isrc = iin(0) * is(0) + iin(1) * is(1) + iin(2) * is(2) + iin(3) * is(3)
              if (idim > 4) {
                idst += i4 * os(4)
                isrc += iin(4) * is(4)
              }
              pout(outOffset + idst) = pin(inOffset + isrc)
              i4 += 1
            }
            i3 += 1
          }
          i2 += 1
        }
        i1 += 1
      }
      i0 += 1
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(input).zero()
    val dT = size(0)
    val dH = size(1)
    val dW = size(2)

    // dims
    val idim = gradInput.dim()
    val xDim = idim - 1
    val yDim = idim - 2
    val zDim = idim - 3

    val isz0 = gradInput.size(1)
    val isz1 = gradInput.size(2)
    val isz2 = gradInput.size(3)
    val isz3 = gradInput.size(4)
    val isz4 = gradInput.size(5)

    val is = gradInput.stride()
    val os = gradOutput.stride()

    val pin = gradInput.storage().array()
    val pout = gradOutput.storage().array()


    val inOffset = gradInput.storageOffset() - 1
    val outOffset = gradOutput.storageOffset() - 1

    // perform the upsampling
    var i0, i1, i2, i3, i4, isrc, idst, x, y, z = 0
    val iin = new Array[Int](5) // Input indices
    val iout = new Array[Int](5) // Output indices

    i0 = 0
    while (i0 < isz0) {
      iout(0) = i0
      iin(0) = i0
      i1 = 0
      while (i1 < isz1) {
        iout(1) = i1
        iin(1) = i1
        i2 = 0
        while (i2 < isz2) {
          iout(2) = i2
          iin(2) = i2
          i3 = 0
          while (i3 < isz3) {
            iout(3) = i3
            iin(3) = i3
            i4 = 0
            while (i4 < isz4) {
              iout(4) = i4
              iin(4) = i4

              idst = i0 * is(0) + i1 * is(1) + i2 * is(2) + i3 * is(3)

              if (idim > 4) {
                idst += i4 * is(4)
              }
              // Now accumulate the gradients from gradOutput
              z = 0
              while (z < dT) {
                y = 0
                while (y < dH) {
                  x = 0
                  while (x < dW) {
                    iout(xDim) = dW * iin(xDim) + x
                    iout(yDim) = dH * iin(yDim) + y
                    iout(zDim) = dT * iin(zDim) + z
                    isrc = iout(0) * os(0) + iout(1) * os(1) + iout(2) * os(2) + iout(3) * os(3)
                    if (idim > 4) {
                      isrc += iout(4) * os(4)
                    }
                    pin(inOffset + idst) = ev.plus(pin(inOffset + idst), pout(outOffset + isrc))
                    x += 1
                  }
                  y += 1
                }
                z += 1
              }
              i4 += 1
            }
            i3 += 1
          }
          i2 += 1
        }
        i1 += 1
      }
      i0 += 1
    }
    gradInput
  }
}

object UpSampling3D {
  def apply[@specialized(Float, Double) T: ClassTag](size: Array[Int])
    (implicit ev: TensorNumeric[T]): UpSampling3D[T] = new UpSampling3D(size)
}
