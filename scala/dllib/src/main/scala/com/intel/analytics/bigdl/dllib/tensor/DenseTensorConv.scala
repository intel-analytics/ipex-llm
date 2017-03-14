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

package com.intel.analytics.bigdl.tensor

import com.intel.analytics.bigdl.tensor.TensorNumericMath._

import scala.reflect.ClassTag

object DenseTensorConv {
  def fullXCorr2Dptr[@specialized(Float, Double) T](output: Storage[T], _outputOffset: Int,
    alpha: T, input: Storage[T], _inputOffset: Int,
    nInputRows: Int, nInputCols: Int, kernel: Storage[T], _kernelOffset: Int,
    nKernelRows: Int, nKernelCols: Int, srow: Int, scol: Int)(
    implicit ev: TensorNumeric[T]): Unit = {
    val nOuputCol = (nInputCols - 1) * scol + nKernelCols
    var inputOffset = _inputOffset
    var yy = 0
    while (yy < nInputRows) {
      var xx = 0
      while (xx < nInputCols) {
        var outputOffset = yy * srow * nOuputCol + xx * scol + _outputOffset
        var kernelOffset = nKernelCols * nKernelRows - 1 + _kernelOffset
        var ky = 0
        while (ky < nKernelRows) {
          val z = input(inputOffset)
          var kx = 0
          while (kx < nKernelCols) {
            output(outputOffset + kx) = ev.plus(output(outputOffset + kx),
              ev.times(z, kernel(kernelOffset - kx)))
            kx += 1
          }
          outputOffset += nOuputCol
          kernelOffset -= nKernelCols
          ky += 1
        }
        inputOffset += 1
        xx += 1
      }
      yy += 1
    }
  }

  def fullConv2Dptr[@specialized(Float, Double) T](output: Storage[T], _outputOffset: Int,
    alpha: T, input: Storage[T], _inputOffset: Int,
    nInputRows: Int, nInputCols: Int, kernel: Storage[T], _kernelOffset: Int,
    nKernelRows: Int, nKernelCols: Int, srow: Int, scol: Int)(
    implicit ev: TensorNumeric[T]): Unit = {
    val nOutputCol = (nInputCols - 1) * scol + nKernelCols
    var inputOffset = _inputOffset
    var yy = 0
    while (yy < nInputRows) {
      var xx = 0
      while (xx < nInputCols) {
        var outputOffset = yy * srow * nOutputCol + xx * scol + _outputOffset
        var kernelOffset = _kernelOffset
        var ky = 0
        while (ky < nKernelRows) {
          val z = ev.times(input(inputOffset), alpha)
          var kx = 0
          while (kx < nKernelCols) {
            output(outputOffset + kx) = ev.plus(output(outputOffset + kx),
              ev.times(z, kernel(kernelOffset + kx)))
            kx += 1
          }
          outputOffset += nOutputCol
          kernelOffset += nKernelCols
          ky += 1
        }
        inputOffset += 1
        xx += 1
      }
      yy += 1
    }
  }

  def validConv2Dptr[@specialized(Float, Double) T](output: Storage[T], _outputOffset: Int,
    alpha: T, input: Storage[T], _inputOffset: Int,
    nInputRows: Int, nInputCols: Int, kernel: Storage[T], _kernelOffset: Int,
    nKernelRows: Int, nKernelCols: Int, srow: Int, scol: Int)(
    implicit ev: TensorNumeric[T]): Unit = {
    val nOutputCol = (nInputCols - nKernelCols) / srow + 1
    val nOutputRow = (nInputRows - nKernelRows) / scol + 1

    var outputOffset = _outputOffset
    var yy = 0
    while (yy < nOutputRow) {
      var xx = 0
      while (xx < nOutputCol) {
        var inputOffset = yy * srow * nInputCols + xx * scol + _inputOffset
        var kernelOffset = nKernelRows * nKernelCols - 1 + _kernelOffset

        var sum = ev.fromType[Int](0)
        var ky = 0
        while (ky < nKernelRows) {
          var kx = 0
          while (kx < nKernelCols) {
            sum = ev.plus(sum, ev.times(input(inputOffset + kx), kernel(kernelOffset - kx)))
            kx += 1
          }
          inputOffset += nInputCols
          kernelOffset -= nKernelCols
          ky += 1
        }
        output(outputOffset) = ev.plus(output(outputOffset), ev.times(alpha, sum))
        outputOffset += 1
        xx += 1
      }
      yy += 1
    }
  }

  def validXCorr2Dptr[@specialized(Float, Double) T](output: Storage[T], _outputOffset: Int,
    alpha: T, input: Storage[T], _inputOffset: Int,
    nInputRows: Int, nInputCols: Int, kernel: Storage[T], _kernelOffset: Int, nKernelRows: Int,
    nKernelCols: Int, srow: Int, scol: Int)(implicit ev: TensorNumeric[T]): Unit = {
    val nOutputCol = (nInputCols - nKernelCols) / srow + 1
    val nOutputRow = (nInputRows - nKernelRows) / scol + 1

    var outputOffset = _outputOffset
    var yy = 0
    while (yy < nOutputRow) {
      var xx = 0
      while (xx < nOutputCol) {
        var inputOffset = yy * srow * nInputCols + xx * scol + _inputOffset
        var kernelOffset = _kernelOffset
        var sum: T = ev.fromType[Int](0)
        var ky = 0
        while (ky < nKernelRows) {
          var kx = 0
          while (kx < nKernelCols) {
            sum = ev.plus(sum, ev.times(input(inputOffset + kx), kernel(kernelOffset + kx)))
            kx += 1
          }
          inputOffset += nInputCols
          kernelOffset += nKernelCols
          ky += 1
        }
        output(outputOffset) = ev.plus(output(outputOffset), ev.times(alpha, sum))
        outputOffset += 1
        xx += 1
      }
      yy += 1
    }
  }


  def conv2d[@specialized(Float, Double) T](output: Storage[T], _outputOffset: Int, alpha: T,
    input: Storage[T], _inputOffset: Int,
    nInputRows: Int, nInputCols: Int, kernel: Storage[T], _kernelOffset: Int, nKernelRows: Int,
    nKernelCols: Int, srow: Int, scol: Int, vf: Char, xc: Char)(
    implicit ev: TensorNumeric[T]): Unit = {
    require(vf == 'F' || vf == 'V', "type of convolution can be 'V' or 'F'")
    require(xc == 'X' || xc == 'C', "type of convolution can be 'X' or 'C'")

    if (vf == 'F') {
      if (xc == 'X') {
        fullXCorr2Dptr(output, _outputOffset, alpha, input, _inputOffset, nInputRows,
          nInputCols, kernel, _kernelOffset, nKernelRows, nKernelCols, srow, scol)
      } else {
        fullConv2Dptr(output, _outputOffset, alpha, input, _inputOffset, nInputRows,
          nInputCols, kernel, _kernelOffset, nKernelRows, nKernelCols, srow, scol)
      }
    } else {
      if (xc == 'X') {
        validXCorr2Dptr(output, _outputOffset, alpha, input, _inputOffset, nInputRows,
          nInputCols, kernel, _kernelOffset, nKernelRows, nKernelCols, srow, scol)
      } else {
        validConv2Dptr(output, _outputOffset, alpha, input, _inputOffset, nInputRows,
          nInputCols, kernel, _kernelOffset, nKernelRows, nKernelCols, srow, scol)
      }
    }

  }

  def validXCorr2DRevptr[@specialized(Float, Double) T](gradWeight: Storage[T],
    _gradWeightOffset: Int, alpha: T, input: Storage[T], _inputOffset: Int,
    ir: Int, ic: Int, output: Storage[T], _outputOffset: Int, kr: Int,
    kc: Int, sr: Int, sc: Int)(implicit ev: TensorNumeric[T]): Unit = {
    val or = ir - (kr - 1) * sr
    val oc = ic - (kc - 1) * sc

    var yy = 0
    while (yy < kr) {
      var xx = 0
      while (xx < kc) {
        var gradWeightOffset = _gradWeightOffset // po
        var inputOffset = _inputOffset + yy * sr * ic + xx * sc // pi
        val z = ev.times(output(_outputOffset + yy * kc + xx), alpha)

        var ky = 0
        while (ky < or) {
          var kx = 0
          while (kx < oc) {
            gradWeight(kx + gradWeightOffset) = ev.plus(gradWeight(kx + gradWeightOffset),
              ev.times(z, input(kx + inputOffset)))
            kx += 1
          }
          gradWeightOffset += oc
          inputOffset += ic
          ky += 1
        }
        xx += 1
      }
      yy += 1
    }
  }

  def conv2Dmul[@specialized(Float, Double) T: ClassTag](alpha: T, t: Tensor[T], k: Tensor[T],
    srow: Int, scol: Int, vf: Char, xc: Char)(implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(t.nDimension() == 2, "input: 2D Tensor expected")
    require(k.nDimension() == 2, "kernel: 2D Tensor expected")

    require(srow >= 1, "Stride should be a positive integer")
    require(scol >= 1, "Stride should be a positive integer")

    val input = t.contiguous()
    val kernel = k.contiguous()

    val nInputRows = input.size(1)
    val nInputCols = input.size(2)

    val nKernelRows = kernel.size(1)
    val nKernelCols = kernel.size(2)

    require((nInputRows >= nKernelRows && nInputCols >= nKernelCols) || vf == 'F',
      "conv2Dmul : Input image is smaller than kernel")

    val nOutputRows = convSize(nInputRows, nKernelRows, srow, vf)
    val nOutputCols = convSize(nInputCols, nKernelCols, scol, vf)

    val result = Tensor[T](nOutputRows, nOutputCols)
    conv2d(result.storage(), result.storageOffset() - 1, alpha, input.storage(),
      input.storageOffset() - 1, nInputRows, nInputCols, kernel.storage(),
      kernel.storageOffset() - 1, nKernelRows, nKernelCols,
      srow, scol, vf, xc)
    result
  }

  def convSize(x: Int, k: Int, s: Int, vf: Char): Int = {
    require(vf == 'F' || vf == 'V', "type of convolution can be 'V' or 'F'")
    if (vf == 'V') {
      (x - k) / s + 1
    } else {
      (x - 1) * s + k
    }
  }

}
