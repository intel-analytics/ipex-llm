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

import java.util

import com.intel.analytics.bigdl.tensor.Tensor

private[nn] object NNPrimitive {
  def im2colDouble(
    fInput: Tensor[Double], input: Tensor[Double],
    kW: Int, kH: Int,
    dW: Int, dH: Int,
    padLeft: Int, padTop: Int, padRight: Int, padBottom: Int,
    outputWidth: Int, outputHeight: Int): Unit = {

    val nInputPlane = input.size(1)
    val inputHeight = input.size(2)
    val inputWidth = input.size(3)

    val inputData = input.storage().array()
    val fInputData = fInput.storage().array()

    var k = 0
    while (k < nInputPlane * kH * kW) {
      val nip = k / (kH * kW)
      val rest = k % (kH * kW)
      val kh = rest / kW
      val kw = rest % kW
      val dstOffset = k * outputHeight * outputWidth + fInput.storageOffset() - 1
      val srcOffset = nip * inputWidth * inputHeight + input.storageOffset() - 1
      if (padLeft > 0 || padRight > 0 || padTop > 0 || padBottom > 0) {
        var y = 0
        while (y < outputHeight) {
          val iy = y * dH - padTop + kh
          if (iy < 0 || iy >= inputHeight) {
            util.Arrays.fill(fInputData, dstOffset + y * outputWidth,
              dstOffset + (y + 1) * outputWidth, 0)
          } else {
            if (dW == 1) {
              val ix = 0 - padLeft + kw
              val lpad = Math.max(0, padLeft - kw)
              val rpad = Math.max(0, padRight - (kW - kw - 1))
              if (outputWidth - rpad - lpad <= 0) {
                util.Arrays.fill(fInputData, dstOffset + y * outputWidth,
                  dstOffset + (y + 1) * outputWidth, 0)
              } else {
                if (lpad > 0) util.Arrays.fill(fInputData, dstOffset + y * outputWidth,
                  dstOffset + y * outputWidth + lpad, 0)
                System.arraycopy(inputData, srcOffset + iy * inputWidth + ix + lpad, fInputData,
                  dstOffset + y * outputWidth + lpad, outputWidth - rpad - lpad)
                if (rpad > 0) util.Arrays.fill(fInputData, dstOffset + (y + 1) * outputWidth - rpad,
                  dstOffset + (y + 1) * outputWidth, 0)
              }
            } else {
              var x = 0
              while (x < outputWidth) {
                val ix = x * dW - padLeft + kw
                if (ix < 0 || ix >= inputWidth) {
                  fInputData(dstOffset + y * outputWidth + x) = 0
                } else {
                  fInputData(dstOffset + y * outputWidth + x) =
                    inputData(srcOffset + iy * inputWidth + ix)
                }
                x += 1
              }
            }
          }
          y += 1
        }
      } else {
        var y = 0
        while (y < outputHeight) {
          val iy = y * dH + kh
          val ix = 0 + kw
          if (dW == 1) {
            System.arraycopy(inputData, srcOffset + iy * inputWidth + ix,
              fInputData, dstOffset + y * outputWidth, outputWidth)
          } else {
            var x = 0
            while (x < outputWidth) {
              fInputData(dstOffset + y * outputWidth + x) =
                inputData(srcOffset + iy * inputWidth + ix + x * dW)
              x += 1
            }
          }
          y += 1
        }
      }
      k += 1
    }
  }

  def im2colFloat(
    fInput: Tensor[Float], input: Tensor[Float],
    kW: Int, kH: Int,
    dW: Int, dH: Int,
    padLeft: Int, padTop: Int, padRight: Int, padBottom: Int,
    outputWidth: Int, outputHeight: Int): Unit = {

    val nInputPlane = input.size(1)
    val inputHeight = input.size(2)
    val inputWidth = input.size(3)

    val inputData = input.storage().array()
    val fInputData = fInput.storage().array()

    var k = 0
    while (k < nInputPlane * kH * kW) {
      val nip = k / (kH * kW)
      val rest = k % (kH * kW)
      val kh = rest / kW
      val kw = rest % kW
      val dstOffset = k * outputHeight * outputWidth + fInput.storageOffset() - 1
      val srcOffset = nip * inputWidth * inputHeight + input.storageOffset() - 1
      if (padLeft > 0 || padRight > 0 || padTop > 0 || padBottom > 0) {
        var y = 0
        while (y < outputHeight) {
          val iy = y * dH - padTop + kh
          if (iy < 0 || iy >= inputHeight) {
            util.Arrays.fill(fInputData, dstOffset + y * outputWidth,
              dstOffset + (y + 1) * outputWidth, 0)
          } else {
            if (dW == 1) {
              val ix = 0 - padLeft + kw
              val lpad = Math.max(0, padLeft - kw)
              val rpad = Math.max(0, padRight - (kW - kw - 1))
              if (outputWidth - rpad - lpad <= 0) {
                util.Arrays.fill(fInputData, dstOffset + y * outputWidth,
                  dstOffset + (y + 1) * outputWidth, 0)
              } else {
                if (lpad > 0) util.Arrays.fill(fInputData, dstOffset + y * outputWidth,
                  dstOffset + y * outputWidth + lpad, 0)
                System.arraycopy(inputData, srcOffset + iy * inputWidth + ix + lpad, fInputData,
                  dstOffset + y * outputWidth + lpad, outputWidth - rpad - lpad)
                if (rpad > 0) util.Arrays.fill(fInputData, dstOffset + (y + 1) * outputWidth - rpad,
                  dstOffset + (y + 1) * outputWidth, 0)
              }
            } else {
              var x = 0
              while (x < outputWidth) {
                val ix = x * dW - padLeft + kw
                if (ix < 0 || ix >= inputWidth) {
                  fInputData(dstOffset + y * outputWidth + x) = 0
                } else {
                  fInputData(dstOffset + y * outputWidth + x) =
                    inputData(srcOffset + iy * inputWidth + ix)
                }
                x += 1
              }
            }
          }
          y += 1
        }
      } else {
        var y = 0
        while (y < outputHeight) {
          val iy = y * dH + kh
          val ix = 0 + kw
          if (dW == 1) {
            System.arraycopy(inputData, srcOffset + iy * inputWidth + ix,
              fInputData, dstOffset + y * outputWidth, outputWidth)
          } else {
            var x = 0
            while (x < outputWidth) {
              fInputData(dstOffset + y * outputWidth + x) =
                inputData(srcOffset + iy * inputWidth + ix + x * dW)
              x += 1
            }
          }
          y += 1
        }
      }
      k += 1
    }
  }

  def col2imDouble(
    fInput: Tensor[Double], input: Tensor[Double],
    kW: Int, kH: Int,
    dW: Int, dH: Int,
    padLeft: Int, padTop: Int, padRight: Int, padBottom: Int,
    outputWidth: Int, outputHeight: Int
  ): Unit = {

    val nInputPlane = input.size(1)
    val inputHeight = input.size(2)
    val inputWidth = input.size(3)

    val inputData = input.storage().array()
    val fInputData = fInput.storage().array()
    var nPlane = 0
    while (nPlane < nInputPlane) {
      var kh = 0
      while (kh < kH) {
        var kw = 0
        while (kw < kW) {
          val srcOffset = nPlane * (kH * kW * outputHeight * outputWidth) +
            kh * (kW * outputHeight * outputWidth) +
            kw * (outputHeight * outputWidth) + fInput.storageOffset() - 1
          val dstOffset = nPlane * (inputHeight * inputWidth) + input.storageOffset() - 1
          if (padLeft > 0 || padRight > 0 || padTop > 0 || padBottom > 0) {
            var y = 0
            while (y < outputHeight) {
              val iy = y * dH - padTop + kh
              if (iy >= 0 && iy < inputHeight) {
                if (dW == 1) {
                  val ix = 0 - padLeft + kw
                  val lPad = Math.max(0, padLeft - kw)
                  val rPad = Math.max(0, padRight - (kW - kw - 1))
                  val inputDataOffset = dstOffset + iy * inputWidth + ix + lPad
                  val fInputDataOffset = srcOffset + y * outputWidth + lPad
                  val n = outputWidth - lPad - rPad
                  var i = 0
                  while (i < n) {
                    inputData(inputDataOffset + i) += fInputData(fInputDataOffset + i)
                    i += 1
                  }
                } else {
                  var x = 0
                  while (x < outputWidth) {
                    val ix = x * dW - padLeft + kw
                    if (ix >= 0 && ix < inputWidth) {
                      inputData(dstOffset + iy * inputWidth + ix) +=
                        fInputData(srcOffset + y * outputWidth + x)
                    }
                    x += 1
                  }
                }
              }
              y += 1
            }
          } else {
            var y = 0
            while (y < outputHeight) {
              val iy = y * dH + kh
              val ix = 0 + kw
              if (dW == 1) {
                var i = 0
                val inputDataOffset = dstOffset + iy * inputWidth + ix
                val fInputDataOffset = srcOffset + y * outputWidth
                while (i < outputWidth) {
                  inputData(inputDataOffset + i) += fInputData(fInputDataOffset + i)
                  i += 1
                }
              } else {
                var x = 0
                while (x < outputWidth) {
                  inputData(dstOffset + iy * inputWidth + ix + x * dW) +=
                    fInputData(srcOffset + y * outputWidth + x)
                  x += 1
                }
              }
              y += 1
            }
          }
          kw += 1
        }
        kh += 1
      }
      nPlane += 1
    }
  }

  def col2imFloat(
    fInput: Tensor[Float], input: Tensor[Float],
    kW: Int, kH: Int,
    dW: Int, dH: Int,
    padLeft: Int, padTop: Int, padRight: Int, padBottom: Int,
    outputWidth: Int, outputHeight: Int
  ): Unit = {

    val nInputPlane = input.size(1)
    val inputHeight = input.size(2)
    val inputWidth = input.size(3)

    val inputData = input.storage().array()
    val fInputData = fInput.storage().array()
    var nPlane = 0
    while (nPlane < nInputPlane) {
      var kh = 0
      while (kh < kH) {
        var kw = 0
        while (kw < kW) {
          val srcOffset = nPlane * (kH * kW * outputHeight * outputWidth) + kh *
            (kW * outputHeight * outputWidth) +
            kw * (outputHeight * outputWidth) + fInput.storageOffset() - 1
          val dstOffset = nPlane * (inputHeight * inputWidth) + input.storageOffset() - 1
          if (padLeft > 0 || padRight > 0 || padTop > 0 || padBottom > 0) {
            var y = 0
            while (y < outputHeight) {
              val iy = y * dH - padTop + kh
              if (iy >= 0 && iy < inputHeight) {
                if (dW == 1) {
                  val ix = 0 - padLeft + kw
                  val lPad = Math.max(0, padLeft - kw)
                  val rPad = Math.max(0, padRight - (kW - kw - 1))
                  val inputDataOffset = dstOffset + iy * inputWidth + ix + lPad
                  val fInputDataOffset = srcOffset + y * outputWidth + lPad
                  val n = outputWidth - lPad - rPad
                  var i = 0
                  while (i < n) {
                    inputData(inputDataOffset + i) += fInputData(fInputDataOffset + i)
                    i += 1
                  }
                } else {
                  var x = 0
                  while (x < outputWidth) {
                    val ix = x * dW - padLeft + kw
                    if (ix >= 0 && ix < inputWidth) {
                      inputData(dstOffset + iy * inputWidth + ix) +=
                        fInputData(srcOffset + y * outputWidth + x)
                    }
                    x += 1
                  }
                }
              }
              y += 1
            }
          } else {
            var y = 0
            while (y < outputHeight) {
              val iy = y * dH + kh
              val ix = 0 + kw
              if (dW == 1) {
                var i = 0
                val inputDataOffset = dstOffset + iy * inputWidth + ix
                val fInputDataOffset = srcOffset + y * outputWidth
                while (i < outputWidth) {
                  inputData(inputDataOffset + i) += fInputData(fInputDataOffset + i)
                  i += 1
                }
              } else {
                var x = 0
                while (x < outputWidth) {
                  inputData(dstOffset + iy * inputWidth + ix + x * dW) +=
                    fInputData(srcOffset + y * outputWidth + x)
                  x += 1
                }
              }
              y += 1
            }
          }
          kw += 1
        }
        kh += 1
      }
      nPlane += 1
    }
  }

  def im2colDoubleNHWC(
    fInput: Tensor[Double], input: Tensor[Double],
    kW: Int, kH: Int,
    dW: Int, dH: Int,
    padLeft: Int, padTop: Int, padRight: Int, padBottom: Int,
    outputWidth: Int, outputHeight: Int): Unit = {

    // padRight and padBottom are used in the NCHW version but not here,
    // add it to keep api consistent

    val nInputPlane = input.size(3)
    val inputHeight = input.size(1)
    val inputWidth = input.size(2)

    val inputData = input.storage().array()
    val fInputData = fInput.storage().array()

    val srcOffset = input.storageOffset() - 1
    val destOffset = fInput.storageOffset() - 1

    var hPad = -padTop
    var fInputCount = 0
    var h = 0
    while (h < outputHeight) {
      var wPad = -padLeft
      var w = 0
      while (w < outputWidth) {
        var ih = hPad
        while (ih < hPad + kH) {
          var iw = wPad
          while(iw < wPad + kW) {
            if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
              val src = srcOffset + (ih * inputWidth + iw) * nInputPlane
              val dest = destOffset + fInputCount
              val n = Math.min(inputWidth, wPad + kW) - iw
              System.arraycopy(inputData, src,
                fInputData, dest, nInputPlane * n)
              fInputCount = fInputCount + nInputPlane * n
              iw = iw + n
            } else {
              val n = if (ih < 0 || ih >= inputHeight || iw >= inputWidth) {
                wPad + kW - iw
              } else {
                0 - iw
              }
              val fromIndex = destOffset + fInputCount
              val toIndex = fromIndex + nInputPlane * n
              util.Arrays.fill(fInputData, fromIndex, toIndex, 0.0)
              fInputCount = fInputCount + nInputPlane * n
              iw = iw + n
            }
          }
          ih = ih + 1
        }
        w = w + 1
        wPad = wPad + dW
      }
      h = h + 1
      hPad = hPad + dH
    }
  }

  def im2colFloatNHWC(
    fInput: Tensor[Float], input: Tensor[Float],
    kW: Int, kH: Int,
    dW: Int, dH: Int,
    padLeft: Int, padTop: Int, padRight: Int, padBottom: Int,
    outputWidth: Int, outputHeight: Int): Unit = {

    // padRight and padBottom are used in the NCHW version but not here,
    // add it to keep api consistent

    val nInputPlane = input.size(3)
    val inputHeight = input.size(1)
    val inputWidth = input.size(2)

    val inputData = input.storage().array()
    val fInputData = fInput.storage().array()

    val srcOffset = input.storageOffset() - 1
    val destOffset = fInput.storageOffset() - 1

    var hPad = -padTop
    var fInputCount = 0
    var h = 0
    while (h < outputHeight) {
      var wPad = -padLeft
      var w = 0
      while (w < outputWidth) {
        var ih = hPad
        while (ih < hPad + kH) {
          var iw = wPad
          while(iw < wPad + kW) {
            if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
              val src = srcOffset + (ih * inputWidth + iw) * nInputPlane
              val dest = destOffset + fInputCount
              val n = Math.min(inputWidth, wPad + kW) - iw
              System.arraycopy(inputData, src,
                fInputData, dest, nInputPlane * n)
              fInputCount = fInputCount + nInputPlane * n
              iw = iw + n
            } else {
              val n = if (ih < 0 || ih >= inputHeight || iw >= inputWidth) {
                wPad + kW - iw
              } else {
                0 - iw
              }
              val fromIndex = destOffset + fInputCount
              val toIndex = fromIndex + nInputPlane * n
              util.Arrays.fill(fInputData, fromIndex, toIndex, 0.0f)
              fInputCount = fInputCount + nInputPlane * n
              iw = iw + n
            }
          }
          ih = ih + 1
        }
        w = w + 1
        wPad = wPad + dW
      }
      h = h + 1
      hPad = hPad + dH
    }
  }

  def col2imDoubleNHWC(
    fInput: Tensor[Double], input: Tensor[Double],
    kW: Int, kH: Int,
    dW: Int, dH: Int,
    padLeft: Int, padTop: Int, padRight: Int, padBottom: Int,
    outputWidth: Int, outputHeight: Int): Unit = {

    // padRight and padBottom are used in the NCHW version but not here,
    // add it to keep api consistent

    val nInputPlane = input.size(3)
    val inputHeight = input.size(1)
    val inputWidth = input.size(2)

    val inputData = input.storage().array()
    val inputOffset = input.storageOffset() - 1
    val fInputData = fInput.storage().array()
    val fInputOffset = fInput.storageOffset() - 1
    var hPad = -padTop
    var h = 0
    var fInputCount = 0
    while (h < outputHeight) {
      var wPad = -padLeft
      var w = 0
      while (w < outputWidth) {
        var ih = hPad
        while (ih < hPad + kH) {
          var iw = wPad
          while (iw < wPad + kW) {
            if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
              val dataImPatch = inputOffset + (ih * inputWidth + iw) * nInputPlane
              var i = 0
              while(i < nInputPlane) {
                inputData(dataImPatch + i) += fInputData(fInputOffset + fInputCount)
                fInputCount = fInputCount + 1
                i = i + 1
              }
            } else {
              fInputCount = fInputCount + nInputPlane
            }
            iw = iw + 1
          }
          ih = ih + 1
        }
        w = w + 1
        wPad = wPad + dW
      }
      h = h + 1
      hPad = hPad + dH
    }
  }

  def col2imFloatNHWC(
    fInput: Tensor[Float], input: Tensor[Float],
    kW: Int, kH: Int,
    dW: Int, dH: Int,
    padLeft: Int, padTop: Int, padRight: Int, padBottom: Int,
    outputWidth: Int, outputHeight: Int): Unit = {

    // padRight and padBottom are used in the NCHW version but not here,
    // add it to keep api consistent

    val nInputPlane = input.size(3)
    val inputHeight = input.size(1)
    val inputWidth = input.size(2)

    val inputData = input.storage().array()
    val inputOffset = input.storageOffset() - 1
    val fInputData = fInput.storage().array()
    val fInputOffset = fInput.storageOffset() - 1
    var hPad = -padTop
    var h = 0
    var fInputCount = 0
    while (h < outputHeight) {
      var wPad = -padLeft
      var w = 0
      while (w < outputWidth) {
        var ih = hPad
        while (ih < hPad + kH) {
          var iw = wPad
          while (iw < wPad + kW) {
            if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
              val dataImPatch = inputOffset + (ih * inputWidth + iw) * nInputPlane
              var i = 0
              while(i < nInputPlane) {
                inputData(dataImPatch + i) += fInputData(fInputOffset + fInputCount)
                fInputCount = fInputCount + 1
                i = i + 1
              }
            } else {
              fInputCount = fInputCount + nInputPlane
            }
            iw = iw + 1
          }
          ih = ih + 1
        }
        w = w + 1
        wPad = wPad + dW
      }
      h = h + 1
      hPad = hPad + dH
    }
  }

  def maxPoolingForwardDouble(
    inputTensor: Tensor[Double],
    outputTensor: Tensor[Double],
    indicesTensor: Tensor[Double],
    oWidth: Int, oHeight: Int,
    kW: Int, kH: Int, dW: Int, dH: Int, padW: Int, padH: Int) {

    val nSlices = inputTensor.size(1)
    val iHeight = inputTensor.size(2)
    val iWidth = inputTensor.size(3)

    val input = inputTensor.storage().array()
    val inputOffset = inputTensor.storageOffset() - 1
    val output = outputTensor.storage().array()
    val outputOffset = outputTensor.storageOffset() - 1
    val indices = indicesTensor.storage().array()
    val indicesOffset = indicesTensor.storageOffset() - 1

    val slices = Range(0, nSlices).iterator
    while (slices.hasNext) {
      val k = slices.next()
      var i = 0
      while (i < oHeight) {
        var j = 0
        while (j < oWidth) {
          // k, i, j output indexers
          var hstart = i * dH - padH
          var wstart = j * dW - padW
          val hend = math.min(hstart + kH, iHeight)
          val wend = math.min(wstart + kW, iWidth)
          hstart = math.max(hstart, 0)
          wstart = math.max(wstart, 0)

          var maxindex = 0  // default is 0
          var maxval = Double.MinValue
          var tcntr = 0
          var y = hstart
          while (y < hend) {
            var x = wstart
            while (x < wend) {
              // k, y, x input indexers
              tcntr = y * iWidth + x
              val value = input(tcntr + inputOffset + k * iWidth * iHeight)
              if (value > maxval) {
                maxval = value
                maxindex = tcntr
              }
              x += 1
            }
            y += 1
          }
          output(outputOffset + k * oWidth * oHeight + i * oWidth + j) = maxval
          indices(indicesOffset + k * oWidth * oHeight + i * oWidth + j) = maxindex + 1
          j += 1
        }
        i += 1
      }
    }
  }

  def maxPoolingForwardFloat(
    inputTensor: Tensor[Float],
    outputTensor: Tensor[Float],
    indicesTensor: Tensor[Float],
    oWidth: Int, oHeight: Int,
    kW: Int, kH: Int, dW: Int, dH: Int, padW: Int, padH: Int) {

    val nSlices = inputTensor.size(1)
    val iHeight = inputTensor.size(2)
    val iWidth = inputTensor.size(3)

    val input = inputTensor.storage().array()
    val inputOffset = inputTensor.storageOffset() - 1
    val output = outputTensor.storage().array()
    val outputOffset = outputTensor.storageOffset() - 1
    val indices = indicesTensor.storage().array()
    val indicesOffset = indicesTensor.storageOffset() - 1

    val slices = Range(0, nSlices).iterator
    while (slices.hasNext) {
      val k = slices.next()
      var i = 0
      while (i < oHeight) {
        var j = 0
        while (j < oWidth) {
          // k, i, j output indexers
          var hstart = i * dH - padH
          var wstart = j * dW - padW
          val hend = math.min(hstart + kH, iHeight)
          val wend = math.min(wstart + kW, iWidth)
          hstart = math.max(hstart, 0)
          wstart = math.max(wstart, 0)

          var maxindex = 0  // default is 0
          var maxval = Float.MinValue
          var tcntr = 0
          var y = hstart
          while (y < hend) {
            var x = wstart
            while (x < wend) {
              // k, y, x input indexers
              tcntr = y * iWidth + x
              val value = input(tcntr + inputOffset + k * iWidth * iHeight)
              if (value > maxval) {
                maxval = value
                maxindex = tcntr
              }
              x += 1
            }
            y += 1
          }
          output(outputOffset + k * oWidth * oHeight + i * oWidth + j) = maxval
          indices(indicesOffset + k * oWidth * oHeight + i * oWidth + j) = maxindex + 1
          j += 1
        }
        i += 1
      }
    }
  }

  def maxPoolingBackwardFloat(
    gradInputTensor: Tensor[Float],
    gradOutputTensor: Tensor[Float],
    indicesTensor: Tensor[Float],
    owidth: Int, oheight: Int): Unit = {

    val nSlices = gradInputTensor.size(1)
    val iHeight = gradInputTensor.size(2)
    val iWidth = gradInputTensor.size(3)

    val gradInput = gradInputTensor.storage().array()
    val gradInputOffset = gradInputTensor.storageOffset() - 1
    val gradOutput = gradOutputTensor.storage().array()
    val gradOutputOffset = gradOutputTensor.storageOffset() - 1
    val indices = indicesTensor.storage().array()
    val indicesOffset = indicesTensor.storageOffset() - 1

    val slices = Range(0, nSlices).iterator
    while (slices.hasNext) {
      val k = slices.next()
      var i = 0
      while (i < oheight) {
        var j = 0
        while (j < owidth) {
          val maxp = indices(i * owidth + j + indicesOffset + k * owidth * oheight).toInt - 1
          gradInput(maxp + k * iWidth * iHeight + gradInputOffset) +=
            gradOutput(gradOutputOffset + k * owidth * oheight + i * owidth + j)
          j += 1
        }
        i += 1
      }
    }
  }

  def maxPoolingBackwardDouble(
    gradInputTensor: Tensor[Double],
    gradOutputTensor: Tensor[Double],
    indicesTensor: Tensor[Double],
    owidth: Int, oheight: Int): Unit = {

    val nSlices = gradInputTensor.size(1)
    val iHeight = gradInputTensor.size(2)
    val iWidth = gradInputTensor.size(3)

    val gradInput = gradInputTensor.storage().array()
    val gradInputOffset = gradInputTensor.storageOffset() - 1
    val gradOutput = gradOutputTensor.storage().array()
    val gradOutputOffset = gradOutputTensor.storageOffset() - 1
    val indices = indicesTensor.storage().array()
    val indicesOffset = indicesTensor.storageOffset() - 1

    val slices = Range(0, nSlices).iterator
    while (slices.hasNext) {
      val k = slices.next()
      var i = 0
      while (i < oheight) {
        var j = 0
        while (j < owidth) {
          val maxp = indices(i * owidth + j + indicesOffset + k * owidth * oheight).toInt - 1
          gradInput(maxp + k * iWidth * iHeight + gradInputOffset) += gradOutput(gradOutputOffset
            + k * owidth * oheight + i * owidth + j)
          j += 1
        }
        i += 1
      }
    }
  }

  def maxPoolingForwardDoubleNHWC(
    inputTensor: Tensor[Double], outputTensor: Tensor[Double], indicesTensor: Tensor[Double],
    oWidth: Int, oHeight: Int,
    kW: Int, kH: Int, dW: Int, dH: Int, padW: Int, padH: Int) {

    val nSlices = inputTensor.size(3)
    val iHeight = inputTensor.size(1)
    val iWidth = inputTensor.size(2)

    val input = inputTensor.storage().array()
    val inputOffset = inputTensor.storageOffset() - 1
    val output = outputTensor.storage().array()
    val outputOffset = outputTensor.storageOffset() - 1
    val indices = indicesTensor.storage().array()
    val indicesOffset = indicesTensor.storageOffset() - 1

    var i = 0
    while (i < oHeight) {
      var j = 0
      var hstart = i * dH - padH
      val hend = math.min(hstart + kH, iHeight)
      hstart = math.max(hstart, 0)
      while (j < oWidth) {
        var wstart = j * dW - padW
        val wend = math.min(wstart + kW, iWidth)
        wstart = math.max(wstart, 0)

        val currOutLocStart = outputOffset + (i * oWidth + j) * nSlices
        val currOutLocEnd = currOutLocStart + nSlices
        val currIndicesLocStart = indicesOffset + (i * oWidth + j) * nSlices
        val currIndicesLocEnd = currIndicesLocStart + nSlices
        util.Arrays.fill(output, currOutLocStart, currOutLocEnd, Double.MinValue)
        util.Arrays.fill(indices, currIndicesLocStart, currIndicesLocEnd, 0)
        var y = hstart
        while (y < hend) {
          var x = wstart
          while (x < wend) {
            // k, y, x input indexers
            val tcntr = y *iWidth + x
            val currInLocStart = inputOffset + tcntr * nSlices
            var n = 0
            while (n < nSlices) {
              val value = input(currInLocStart + n)
              if (value > output(currOutLocStart + n)) {
                output(currOutLocStart + n) = value
                indices(currOutLocStart + n) = tcntr + 1
              }
              n = n + 1
            }
            x += 1
          }
          y += 1
        }
        j += 1
      }
      i += 1
    }
  }

  def maxPoolingForwardFloatNHWC(
    inputTensor: Tensor[Float], outputTensor: Tensor[Float], indicesTensor: Tensor[Float],
    oWidth: Int, oHeight: Int,
    kW: Int, kH: Int, dW: Int, dH: Int, padW: Int, padH: Int) {

    val nSlices = inputTensor.size(3)
    val iHeight = inputTensor.size(1)
    val iWidth = inputTensor.size(2)

    val input = inputTensor.storage().array()
    val inputOffset = inputTensor.storageOffset() - 1
    val output = outputTensor.storage().array()
    val outputOffset = outputTensor.storageOffset() - 1
    val indices = indicesTensor.storage().array()
    val indicesOffset = indicesTensor.storageOffset() - 1

    var i = 0
    while (i < oHeight) {
      var j = 0
      var hstart = i * dH - padH
      val hend = math.min(hstart + kH, iHeight)
      hstart = math.max(hstart, 0)
      while (j < oWidth) {
        var wstart = j * dW - padW
        val wend = math.min(wstart + kW, iWidth)
        wstart = math.max(wstart, 0)

        val currOutLocStart = outputOffset + (i * oWidth + j) * nSlices
        val currOutLocEnd = currOutLocStart + nSlices
        val currIndicesLocStart = indicesOffset + (i * oWidth + j) * nSlices
        val currIndicesLocEnd = currIndicesLocStart + nSlices
        util.Arrays.fill(output, currOutLocStart, currOutLocEnd, Float.MinValue)
        util.Arrays.fill(indices, currIndicesLocStart, currIndicesLocEnd, 0)
        var y = hstart
        while (y < hend) {
          var x = wstart
          while (x < wend) {
            // k, y, x input indexers
            val tcntr = y *iWidth + x
            val currInLocStart = inputOffset + tcntr * nSlices
            var n = 0
            while (n < nSlices) {
              val value = input(currInLocStart + n)
              if (value > output(currOutLocStart + n)) {
                output(currOutLocStart + n) = value
                indices(currOutLocStart + n) = tcntr + 1
              }
              n = n + 1
            }
            x += 1
          }
          y += 1
        }
        j += 1
      }
      i += 1
    }
  }

  def maxPoolingBackwardDoubleNHWC(
    gradInputTensor: Tensor[Double],
    gradOutputTensor: Tensor[Double],
    indicesTensor: Tensor[Double],
    oWidth: Int, oHeight: Int): Unit = {

    val nSlices = gradInputTensor.size(3)
    val iHeight = gradInputTensor.size(1)
    val iWidth = gradInputTensor.size(2)

    val gradInput = gradInputTensor.storage().array()
    val gradInputOffset = gradInputTensor.storageOffset() - 1
    val gradOutput = gradOutputTensor.storage().array()
    val gradOutputOffset = gradOutputTensor.storageOffset() - 1
    val indices = indicesTensor.storage().array()
    val indicesOffset = indicesTensor.storageOffset() - 1

    var i = 0
    while (i < oHeight) {
      var j = 0
      while (j < oWidth) {
        val currOutLocStart = gradOutputOffset + (i * oWidth + j) * nSlices
        val currIndicesLocStart = indicesOffset + (i * oWidth + j) * nSlices
        var n = 0
        while (n < nSlices) {
          val maxIndex = indices(currIndicesLocStart + n).toInt - 1
          val grad = gradOutput(currOutLocStart + n)
          gradInput(gradInputOffset + maxIndex * nSlices + n) += grad
          n = n + 1
        }
        j += 1
      }
      i += 1
    }
  }

  def maxPoolingBackwardFloatNHWC(
    gradInputTensor: Tensor[Float],
    gradOutputTensor: Tensor[Float],
    indicesTensor: Tensor[Float],
    oWidth: Int, oHeight: Int): Unit = {

    val nSlices = gradInputTensor.size(3)
    val iHeight = gradInputTensor.size(1)
    val iWidth = gradInputTensor.size(2)

    val gradInput = gradInputTensor.storage().array()
    val gradInputOffset = gradInputTensor.storageOffset() - 1
    val gradOutput = gradOutputTensor.storage().array()
    val gradOutputOffset = gradOutputTensor.storageOffset() - 1
    val indices = indicesTensor.storage().array()
    val indicesOffset = indicesTensor.storageOffset() - 1

    var i = 0
    while (i < oHeight) {
      var j = 0
      while (j < oWidth) {
        val currOutLocStart = gradOutputOffset + (i * oWidth + j) * nSlices
        val currIndicesLocStart = indicesOffset + (i * oWidth + j) * nSlices
        var n = 0
        while (n < nSlices) {
          val maxIndex = indices(currIndicesLocStart + n).toInt - 1
          val grad = gradOutput(currOutLocStart + n)
          gradInput(gradInputOffset + maxIndex * nSlices + n) += grad
          n = n + 1
        }
        j += 1
      }
      i += 1
    }
  }


  def temporalMaxPoolingBackwardDouble(
    gradInput: Array[Double], gradInputOffset: Int,
    gradOutput: Array[Double], gradOutputOffset: Int,
    indices: Array[Double], indicesOffset: Int,
    nSlices: Int, frameSize: Int,
    kW: Int, dW: Int): Unit = {
    for (t <- Range(0, nSlices)) {
      val gip = gradInputOffset + t * frameSize * dW
      val gop = gradOutputOffset + t * frameSize
      val xp = indicesOffset + t * frameSize

      var y = 0
      while (y < frameSize) {
        val maxIndex = indices(xp + y).toInt - 1
        if (maxIndex != -1) {
          gradInput(gip + maxIndex * frameSize + y) +=
            gradOutput(gop + y)
        }
        y += 1
      }
    }
  }

  def temporalMaxPoolingBackwardFloat(
    gradInput: Array[Float], gradInputOffset: Int,
    gradOutput: Array[Float], gradOutputOffset: Int,
    indices: Array[Float], indicesOffset: Int,
    nSlices: Int, frameSize: Int,
    kW: Int, dW: Int): Unit = {
    for (t <- Range(0, nSlices)) {
      val gip = gradInputOffset + t * frameSize * dW
      val gop = gradOutputOffset + t * frameSize
      val xp = indicesOffset + t * frameSize

      var y = 0
      while (y < frameSize) {
        val maxIndex = indices(xp + y).toInt - 1
        if (maxIndex != -1) {
          gradInput(gip + maxIndex * frameSize + y) +=
            gradOutput(gop + y)
        }
        y += 1
      }
    }
  }

  def temporalMaxPoolingForwardDouble(
    input: Array[Double], inputOffset: Int,
    output: Array[Double], outputOffset: Int,
    indices: Array[Double], indicesOffset: Int,
    nSlices: Int, frameSize: Int,
    kW: Int, dW: Int): Unit = {
    val slices = Range(0, nSlices).iterator
    while (slices.hasNext) {
      val t = slices.next()
      val ip = inputOffset + t * frameSize * dW
      val op = outputOffset + t * frameSize
      val xp = indicesOffset + t * frameSize
      var y = 0
      while (y < frameSize) {
        var maxindex = 0  // default is 0
        var maxval = Double.MinValue
        var x = 0
        while (x < kW) {
          val value = input(ip + x * frameSize + y)
          if (value > maxval) {
            maxval = value
            maxindex = x
          }
          x += 1
        }
        output(op + y) = maxval
        indices(xp + y) = maxindex + 1
        y += 1
      }
    }
  }

  def temporalMaxPoolingForwardFloat(
    input: Array[Float], inputOffset: Int,
    output: Array[Float], outputOffset: Int,
    indices: Array[Float], indicesOffset: Int,
    nSlices: Int, frameSize: Int,
    kW: Int, dW: Int): Unit = {
    val slices = Range(0, nSlices).iterator
    while (slices.hasNext) {
      val t = slices.next()
      val ip = inputOffset + t * frameSize * dW
      val op = outputOffset + t * frameSize
      val xp = indicesOffset + t * frameSize
      var y = 0
      while (y < frameSize) {
        var maxindex = 0  // default is 0
        var maxval = Float.MinValue
        var x = 0
        while (x < kW) {
          val value = input(ip + x * frameSize + y)
          if (value > maxval) {
            maxval = value
            maxindex = x
          }
          x += 1
        }
        output(op + y) = maxval
        indices(xp + y) = maxindex + 1
        y += 1
      }
    }
  }

  // For SpatialFullConvolution
  def col2imWithDilationDouble(columns : Tensor[Double], image : Tensor[Double],
    channels : Int, height : Int, width : Int,
    kernelH : Int, kernelW : Int,
    padH : Int, padW : Int,
    strideH : Int, strideW : Int,
    dilationH : Int, dilationW : Int) {

    val dataIm = image.storage().array()
    val dataImOffset = image.storageOffset() - 1
    val dataCol = columns.storage().array()
    val dataColOffset = columns.storageOffset() - 1

    val heightCol = (height + 2 * padH -
      (dilationH * (kernelH - 1) + 1)) / strideH + 1
    val widthCol = (width + 2 * padW -
      (dilationW * (kernelW - 1) + 1)) / strideW + 1
    val channelsCol = channels * kernelH * kernelW
    var cCol = 0
    while (cCol < channelsCol) {
      val wOffset = cCol % kernelW
      val hOffset = (cCol / kernelW) % kernelH
      val cIm = cCol / kernelH / kernelW
      var hCol = 0
      while (hCol < heightCol) {
        var wCol = 0
        while (wCol < widthCol) {
          val hIm = hCol * strideH - padH + hOffset * dilationH
          val wIm = wCol * strideW - padW + wOffset * dilationW
          if (hIm >= 0 && hIm < height && wIm >= 0 && wIm < width) {
            dataIm((cIm * height + hIm) * width + wIm + dataImOffset) +=
              dataCol((cCol * heightCol + hCol) * widthCol + wCol + dataColOffset)
          }
          wCol += 1
        }
        hCol += 1
      }
      cCol += 1
    }
  }

  def col2imWithDilationFloat(columns : Tensor[Float], image : Tensor[Float],
    channels : Int, height : Int, width : Int,
    kernelH : Int, kernelW : Int,
    padH : Int, padW : Int,
    strideH : Int, strideW : Int,
    dilationH : Int, dilationW : Int) {

    val dataIm = image.storage().array()
    val dataImOffset = image.storageOffset() - 1
    val dataCol = columns.storage().array()
    val dataColOffset = columns.storageOffset() - 1

    val heightCol = (height + 2 * padH -
      (dilationH * (kernelH - 1) + 1)) / strideH + 1
    val widthCol = (width + 2 * padW -
      (dilationW * (kernelW - 1) + 1)) / strideW + 1
    val channelsCol = channels * kernelH * kernelW
    var cCol = 0
    while (cCol < channelsCol) {
      val wOffset = cCol % kernelW
      val hOffset = (cCol / kernelW) % kernelH
      val cIm = cCol / kernelH / kernelW
      var hCol = 0
      while (hCol < heightCol) {
        var wCol = 0
        while (wCol < widthCol) {
          val hIm = hCol * strideH - padH + hOffset * dilationH
          val wIm = wCol * strideW - padW + wOffset * dilationW
          if (hIm >= 0 && hIm < height && wIm >= 0 && wIm < width) {
            dataIm((cIm * height + hIm) * width + wIm + dataImOffset) +=
              dataCol((cCol * heightCol + hCol) * widthCol + wCol + dataColOffset)
          }
          wCol += 1
        }
        hCol += 1
      }
      cCol += 1
    }
  }

  def im2colWithDilationDouble(image: Tensor[Double], columns: Tensor[Double],
    channels : Int, height : Int, width : Int,
    kernelH : Int, kernelW : Int,
    padH : Int, padW : Int,
    strideH : Int, strideW : Int,
    dilationH : Int, dilationW : Int): Unit = {

    val dataIm = image.storage().array()
    val dataImOffset = image.storageOffset() - 1
    val dataCol = columns.storage().array()
    val dataColOffset = columns.storageOffset() - 1

    val heightCol = (height + 2 * padH -
      (dilationH * (kernelH - 1) + 1)) / strideH + 1
    val widthCol = (width + 2 * padW -
      (dilationW * (kernelW - 1) + 1)) / strideW + 1
    val channelsCol = channels * kernelH * kernelW
    var cCol = 0
    while (cCol < channelsCol) {
      val wOffset = cCol % kernelW
      val hOffset = (cCol / kernelW) % kernelH
      val cIm = cCol / kernelH / kernelW
      var hCol = 0
      while (hCol < heightCol) {
        var wCol = 0
        while (wCol < widthCol) {
          val hIm = hCol * strideH - padH + hOffset * dilationH
          val wIm = wCol * strideW - padW + wOffset * dilationW
          dataCol((cCol * heightCol + hCol) * widthCol + wCol + dataColOffset) =
            if (hIm >= 0 && wIm >= 0 && hIm < height && wIm < width) {
              dataIm((cIm * height + hIm) * width + wIm + dataImOffset)
            }
            else {
              0
            }
          wCol += 1
        }
        hCol += 1
      }
      cCol += 1
    }
  }

  def im2colWithDilationFloat(image: Tensor[Float], columns: Tensor[Float],
    channels : Int, height : Int, width : Int,
    kernelH : Int, kernelW : Int,
    padH : Int, padW : Int,
    strideH : Int, strideW : Int,
    dilationH : Int, dilationW : Int): Unit = {

    val dataIm = image.storage().array()
    val dataImOffset = image.storageOffset() - 1
    val dataCol = columns.storage().array()
    val dataColOffset = columns.storageOffset() - 1

    val heightCol = (height + 2 * padH -
      (dilationH * (kernelH - 1) + 1)) / strideH + 1
    val widthCol = (width + 2 * padW -
      (dilationW * (kernelW - 1) + 1)) / strideW + 1
    val channelsCol = channels * kernelH * kernelW
    var cCol = 0
    while (cCol < channelsCol) {
      val wOffset = cCol % kernelW
      val hOffset = (cCol / kernelW) % kernelH
      val cIm = cCol / kernelH / kernelW
      var hCol = 0
      while (hCol < heightCol) {
        var wCol = 0
        while (wCol < widthCol) {
          val hIm = hCol * strideH - padH + hOffset * dilationH
          val wIm = wCol * strideW - padW + wOffset * dilationW
          dataCol((cCol * heightCol + hCol) * widthCol + wCol + dataColOffset) =
            if (hIm >= 0 && wIm >= 0 && hIm < height && wIm < width) {
              dataIm((cIm * height + hIm) * width + wIm + dataImOffset)
            }
            else {
              0
            }
          wCol += 1
        }
        hCol += 1
      }
      cCol += 1
    }
  }

  def unfoldedCopyVolDouble(fInput: Tensor[Double], input: Tensor[Double],
    kT: Int, kW: Int, kH: Int,
    dT: Int, dW: Int, dH: Int,
    padFront: Int, padLeft: Int, padTop: Int,
    padBack: Int, padRight: Int, padBottom: Int,
    nInputPlane: Int,
    inputDepth: Int, inputWidth: Int, inputHeight: Int, outputDepth: Int,
    outputWidth: Int, outputHeight: Int): Unit = {
    val inputData = input.storage().array()
    val fInputData = fInput.storage().array()

    var k = 0
    while (k < nInputPlane * kT * kH * kW) {
      val nip = k / (kT * kH * kW)
      var rest = k % (kT * kH * kW)
      val kt = rest / (kH * kW)
      rest = rest % (kH * kW)
      val kh = rest / kW
      val kw = rest % kW
      var t, x, y, it, ix, iy = 0
      val dstOffset = nip * (kT * kH * kW * outputDepth * outputHeight * outputWidth) +
        kt * (kH * kW * outputDepth * outputHeight * outputWidth) +
        kh * (kW * outputDepth * outputHeight * outputWidth) +
        kw * (outputDepth * outputHeight * outputWidth) + fInput.storageOffset() - 1
      val srcOffset = nip * (inputDepth * inputHeight * inputWidth) + input.storageOffset() - 1

      if (padFront > 0 || padBack > 0 || padLeft > 0 || padRight > 0 ||
        padBottom > 0 || padTop > 0) {
        t = 0
        while (t < outputDepth) {
          it = t * dT - padFront + kt
          var y = 0
          while (y < outputHeight) {
            iy = y * dH - padTop + kh
            x = 0
            while (x < outputWidth) {
              ix = x * dW - padLeft + kw
              if (it < 0 || it >= inputDepth || iy < 0 || iy >= inputHeight ||
                ix < 0 || ix >= inputWidth) {
                fInputData(dstOffset + t * outputHeight * outputWidth + y * outputWidth + x) = 0
              } else {
                fInputData(dstOffset + t * outputHeight * outputWidth + y * outputWidth + x)
                  = inputData(srcOffset + it * inputHeight * inputWidth + iy * inputWidth + ix)
              }
              x += 1
            }
            y += 1
          }
          t += 1
        }
      } else {
        t = 0
        while (t < outputDepth) {
          it = t * dT + kt
          y = 0
          while (y < outputHeight) {
            iy = y * dH + kh
            x = 0
            while (x < outputWidth) {
              ix = x * dW + kw
              fInputData(dstOffset + t * outputHeight * outputWidth + y * outputWidth + x)
                = inputData(srcOffset + it * inputHeight * inputWidth + iy * inputWidth + ix)
              x += 1
            }
            y += 1
          }
          t += 1
        }
      }
      k += 1
    }
  }

  def unfoldedCopyVolFloat(fInput: Tensor[Float], input: Tensor[Float],
    kT: Int, kW: Int, kH: Int,
    dT: Int, dW: Int, dH: Int,
    padFront: Int, padLeft: Int, padTop: Int,
    padBack: Int, padRight: Int, padBottom: Int,
    nInputPlane: Int,
    inputDepth: Int, inputWidth: Int, inputHeight: Int, outputDepth: Int,
    outputWidth: Int, outputHeight: Int): Unit = {
    val inputData = input.storage().array()
    val fInputData = fInput.storage().array()

    var k = 0
    while (k < nInputPlane * kT * kH * kW) {
      val nip = k / (kT * kH * kW)
      var rest = k % (kT * kH * kW)
      val kt = rest / (kH * kW)
      rest = rest % (kH * kW)
      val kh = rest / kW
      val kw = rest % kW
      var t, x, y, it, ix, iy = 0
      val dstOffset = nip * (kT * kH * kW * outputDepth * outputHeight * outputWidth) +
        kt * (kH * kW * outputDepth * outputHeight * outputWidth) +
        kh * (kW * outputDepth * outputHeight * outputWidth) +
        kw * (outputDepth * outputHeight * outputWidth) + fInput.storageOffset() - 1
      val srcOffset = nip * (inputDepth * inputHeight * inputWidth) + input.storageOffset() - 1

      if (padFront > 0 || padLeft > 0 || padTop > 0 || padBack > 0
        || padRight > 0 || padBottom > 0) {
        t = 0
        while (t < outputDepth) {
          it = t * dT - padFront + kt
          var y = 0
          while (y < outputHeight) {
            iy = y * dH - padTop + kh
            x = 0
            while (x < outputWidth) {
              ix = x * dW - padLeft + kw
              if (it < 0 || it >= inputDepth || iy < 0 || iy >= inputHeight ||
                ix < 0 || ix >= inputWidth) {
                fInputData(dstOffset + t * outputHeight * outputWidth + y * outputWidth + x) = 0f
              } else {
                fInputData(dstOffset + t * outputHeight * outputWidth + y * outputWidth + x)
                  = inputData(srcOffset + it * inputHeight * inputWidth + iy * inputWidth + ix)
              }
              x += 1
            }
            y += 1
          }
          t += 1
        }
      } else {
        t = 0
        while (t < outputDepth) {
          it = t * dT + kt
          y = 0
          while (y < outputHeight) {
            iy = y * dH + kh
            x = 0
            while (x < outputWidth) {
              ix = x * dW + kw
              fInputData(dstOffset + t * outputHeight * outputWidth + y * outputWidth + x)
                = inputData(srcOffset + it * inputHeight * inputWidth + iy * inputWidth + ix)
              x += 1
            }
            y += 1
          }
          t += 1
        }
      }
      k += 1
    }
  }

  def unfoldedAccVolDouble(fInput: Tensor[Double], input: Tensor[Double], kT: Int, kW: Int, kH: Int,
    dT: Int, dW: Int, dH: Int,
    padFront: Int, padLeft: Int, padTop: Int,
    padBack: Int, padRight: Int, padBottom: Int,
    nInputPlane: Int, inputDepth: Int,
    inputWidth: Int, inputHeight: Int,
    outputDepth: Int, outputWidth: Int, outputHeight: Int): Unit = {
    var nip, kt, kw, kh, t, y, x, it, ix, iy = 0
    val inputData = input.storage().array()
    val fInputData = fInput.storage().array()
    nip = 0
    while (nip < nInputPlane) {
      kt = 0
      while (kt < kT) {
        kh = 0
        while (kh < kH) {
          kw = 0
          while (kw < kW) {
            val srcOffset = nip * (kT * kH * kW * outputDepth * outputHeight * outputWidth) +
              kt * (kH * kW * outputDepth * outputHeight * outputWidth) +
              kh * (kW * outputDepth * outputHeight * outputWidth) +
              kw * (outputDepth * outputHeight * outputWidth) + fInput.storageOffset() - 1

            val dstOffset = nip * (inputDepth * inputHeight * inputWidth) +
              input.storageOffset() - 1
            if (padFront > 0 || padLeft > 0 || padTop > 0 || padBack > 0
              || padRight > 0 || padBottom > 0) {
              t = 0
              while (t < outputDepth) {
                it = t * dT - padFront + kt
                y = 0
                while (y < outputHeight) {
                  iy = y * dH - padTop + kh
                  x = 0
                  while (x < outputWidth) {
                    ix = x * dW - padLeft + kw
                    if (it < 0 || it >= inputDepth || iy < 0 || iy >= inputHeight ||
                      ix < 0 || ix >= inputWidth) {

                    }
                    else {
                      inputData(dstOffset + it * inputHeight * inputWidth + iy * inputWidth + ix) +=
                          fInputData(srcOffset + t * outputHeight * outputWidth +
                            y * outputWidth + x)
                    }
                    x += 1
                  }
                  y += 1
                }
                t += 1
              }
            }
            else {
              t = 0
              while (t < outputDepth) {
                it = t * dT + kt
                y = 0
                while (y < outputHeight) {
                  iy = y * dH + kh
                  x = 0
                  while (x < outputWidth) {
                    ix = x * dW + kw
                    inputData(dstOffset + it * inputHeight * inputWidth + iy * inputWidth + ix) +=
                      fInputData(srcOffset + t * outputHeight * outputWidth + y * outputWidth + x)
                    x += 1
                  }
                  y += 1
                }
                t += 1
              }
            }
            kw += 1
          }
          kh += 1
        }
        kt += 1
      }
      nip += 1
    }
  }

  def unfoldedAccVolFloat(fInput: Tensor[Float], input: Tensor[Float], kT: Int, kW: Int, kH: Int,
    dT: Int, dW: Int, dH: Int,
    padFront: Int, padLeft: Int, padTop: Int,
    padBack: Int, padRight: Int, padBottom: Int,
    nInputPlane: Int, inputDepth: Int,
    inputWidth: Int, inputHeight: Int,
    outputDepth: Int, outputWidth: Int, outputHeight: Int): Unit = {
    var nip, kt, kw, kh, t, y, x, it, ix, iy = 0
    val inputData = input.storage().array()
    val fInputData = fInput.storage().array()
    nip = 0
    while (nip < nInputPlane) {
      kt = 0
      while (kt < kT) {
        kh = 0
        while (kh < kH) {
          kw = 0
          while (kw < kW) {
            val srcOffset = nip * (kT * kH * kW * outputDepth * outputHeight * outputWidth) +
              kt * (kH * kW * outputDepth * outputHeight * outputWidth) +
              kh * (kW * outputDepth * outputHeight * outputWidth) +
              kw * (outputDepth * outputHeight * outputWidth) + fInput.storageOffset() - 1

            val dstOffset = nip * (inputDepth * inputHeight * inputWidth) +
              input.storageOffset() - 1
            if (padFront > 0 || padLeft > 0 || padTop > 0 || padBack > 0
              || padRight > 0 || padBottom > 0) {
              t = 0
              while (t < outputDepth) {
                it = t * dT - padFront + kt
                y = 0
                while (y < outputHeight) {
                  iy = y * dH - padTop + kh
                  x = 0
                  while (x < outputWidth) {
                    ix = x * dW - padLeft + kw
                    if (it < 0 || it >= inputDepth || iy < 0 || iy >= inputHeight ||
                      ix < 0 || ix >= inputWidth) {

                    }
                    else {
                      inputData(dstOffset + it * inputHeight * inputWidth + iy * inputWidth + ix) +=
                        fInputData(srcOffset + t * outputHeight * outputWidth +
                          y * outputWidth + x)
                    }
                    x += 1
                  }
                  y += 1
                }
                t += 1
              }
            }
            else {
              t = 0
              while (t < outputDepth) {
                it = t * dT + kt
                y = 0
                while (y < outputHeight) {
                  iy = y * dH + kh
                  x = 0
                  while (x < outputWidth) {
                    ix = x * dW + kw
                    inputData(dstOffset + it * inputHeight * inputWidth + iy * inputWidth + ix) +=
                      fInputData(srcOffset + t * outputHeight * outputWidth + y * outputWidth + x)
                    x += 1
                  }
                  y += 1
                }
                t += 1
              }
            }
            kw += 1
          }
          kh += 1
        }
        kt += 1
      }
      nip += 1
    }
  }

  def vol2colDouble(
    vol: Tensor[Double],
    channels: Int,
    depth: Int, height: Int, width: Int,
    kT: Int, kH: Int, kW: Int,
    pT: Int, pH: Int, pW: Int,
    dT: Int, dH: Int, dW: Int,
    dilationT: Int, dilationH: Int, dilationW: Int,
    col: Tensor[Double]
  ): Unit = {
    val colData = col.storage().array()
    val colDataOffset = col.storageOffset() - 1
    val volData = vol.storage().array()
    val volDataOffset = vol.storageOffset() - 1
    val depthCol = (depth + 2 * pT - (dilationT * (kT - 1) + 1)) / dT + 1
    val widthCol = (width + 2 * pW - (dilationW * (kW - 1) + 1)) / dW + 1
    val heightCol = (height + 2 * pH - (dilationH * (kH - 1) + 1)) / dH + 1
    val channelsCol = channels * kT * kW * kH

    var c = 0
    while (c < channelsCol) {
      val wOffset = c % kW
      val hOffset = (c / kW) % kH
      val tOffset = (c / kW / kH) % kT
      val cVol = c / kT / kH / kW

      var t = 0
      while (t < depthCol) {
        var h = 0
        while (h < heightCol) {
          var w = 0
          while (w < widthCol) {
            val tPad = t * dT - pT + tOffset * dilationT
            val hPad = h * dH - pH + hOffset * dilationH
            val wPad = w * dW - pW + wOffset * dilationW

            if (tPad >= 0 && tPad < depth &&
              hPad >= 0 && hPad < height &&
              wPad >= 0 && wPad < width) {
              colData(((c * depthCol + t) * heightCol + h) * widthCol + w + colDataOffset) =
                volData(((cVol * depth + tPad) * height + hPad) * width + wPad + volDataOffset)
            } else {
              colData(((c * depthCol + t) * heightCol + h) * widthCol + w + colDataOffset) = 0.0
            }
            w += 1
          }
          h += 1
        }
        t += 1
      }
      c += 1
    }
  }

  def vol2colFloat(
    vol: Tensor[Float],
    channels: Int,
    depth: Int, height: Int, width: Int,
    kT: Int, kH: Int, kW: Int,
    pT: Int, pH: Int, pW: Int,
    dT: Int, dH: Int, dW: Int,
    dilationT: Int, dilationH: Int, dilationW: Int,
    col: Tensor[Float]
  ): Unit = {
    val colData = col.storage().array()
    val colDataOffset = col.storageOffset() - 1
    val volData = vol.storage().array()
    val volDataOffset = vol.storageOffset() - 1
    val depthCol = (depth + 2 * pT - (dilationT * (kT - 1) + 1)) / dT + 1
    val widthCol = (width + 2 * pW - (dilationW * (kW - 1) + 1)) / dW + 1
    val heightCol = (height + 2 * pH - (dilationH * (kH - 1) + 1)) / dH + 1
    val channelsCol = channels * kT * kW * kH

    var c = 0
    while (c < channelsCol) {
      val wOffset = c % kW
      val hOffset = (c / kW) % kH
      val tOffset = (c / kW / kH) % kT
      val cVol = c / kT / kH / kW

      var t = 0
      while (t < depthCol) {
        var h = 0
        while (h < heightCol) {
          var w = 0
          while (w < widthCol) {
            val tPad = t * dT - pT + tOffset * dilationT
            val hPad = h * dH - pH + hOffset * dilationH
            val wPad = w * dW - pW + wOffset * dilationW

            if (tPad >= 0 && tPad < depth &&
              hPad >= 0 && hPad < height &&
              wPad >= 0 && wPad < width) {
              colData(((c * depthCol + t) * heightCol + h) * widthCol + w + colDataOffset) =
                volData(((cVol * depth + tPad) * height + hPad) * width + wPad + volDataOffset)
            } else {
              colData(((c * depthCol + t) * heightCol + h) * widthCol + w + colDataOffset) = 0f
            }
            w += 1
          }
          h += 1
        }
        t += 1
      }
      c += 1
    }
  }

  def col2volDouble(
    col: Tensor[Double],
    channels: Int,
    depth: Int, height: Int, width: Int,
    kT: Int, kH: Int, kW: Int,
    pT: Int, pH: Int, pW: Int,
    dT: Int, dH: Int, dW: Int,
    dilationT: Int, dilationH: Int, dilationW: Int,
    vol: Tensor[Double]
  ): Unit = {
    val colData = col.storage().array()
    val colDataOffset = col.storageOffset() - 1
    val volData = vol.storage().array()
    val volDataOffset = vol.storageOffset() - 1

    val depthCol = (depth + 2 * pT - (dilationT * (kT - 1) + 1)) / dT + 1
    val heightCol = (height + 2 * pH - (dilationH * (kH - 1) + 1)) / dH + 1
    val widthCol = (width + 2 * pW - (dilationW * (kW - 1) + 1)) / dW + 1
    val channelsCol = channels * kT * kW * kH

    var c = 0
    while (c < channelsCol) {
      val wOffset = c % kW
      val hOffset = (c / kW) % kH
      val tOffset = (c / kW / kH) % kT
      val cVol = c / kT / kH / kW

      var t = 0
      while (t < depthCol) {
        var h = 0
        while (h < heightCol) {
          var w = 0
          while (w < widthCol) {
            val tPad = t * dT - pT + tOffset * dilationT
            val hPad = h * dH - pH + hOffset * dilationH
            val wPad = w * dW - pW + wOffset * dilationW

            if (tPad >= 0 && tPad < depth &&
              hPad >= 0 && hPad < height &&
              wPad >= 0 && wPad < width) {
              volData(((cVol * depth + tPad) * height + hPad) * width + wPad + volDataOffset) +=
                colData(((c * depthCol + t) * heightCol + h) * widthCol + w + colDataOffset)
            }
            w += 1
          }
          h += 1
        }
        t += 1
      }
      c += 1
    }
  }

  def col2volFloat(
    col: Tensor[Float],
    channels: Int,
    depth: Int, width: Int, height: Int,
    kT: Int, kW: Int, kH: Int,
    pT: Int, pW: Int, pH: Int,
    dT: Int, dW: Int, dH: Int,
    dilationT: Int, dilationW: Int, dilationH: Int,
    vol: Tensor[Float]
  ): Unit = {
    val colData = col.storage().array()
    val colDataOffset = col.storageOffset() - 1
    val volData = vol.storage().array()
    val volDataOffset = vol.storageOffset() - 1

    val depthCol = (depth + 2 * pT - (dilationT * (kT - 1) + 1)) / dT + 1
    val heightCol = (height + 2 * pH - (dilationH * (kH - 1) + 1)) / dH + 1
    val widthCol = (width + 2 * pW - (dilationW * (kW - 1) + 1)) / dW + 1
    val channelsCol = channels * kT * kW * kH

    var c = 0
    while (c < channelsCol) {
      val wOffset = c % kW
      val hOffset = (c / kW) % kH
      val tOffset = (c / kW / kH) % kT
      val cVol = c / kT / kH / kW

      var t = 0
      while (t < depthCol) {
        var h = 0
        while (h < heightCol) {
          var w = 0
          while (w < widthCol) {
            val tPad = t * dT - pT + tOffset * dilationT
            val hPad = h * dH - pH + hOffset * dilationH
            val wPad = w * dW - pW + wOffset * dilationW

            if (tPad >= 0 && tPad < depth &&
              hPad >= 0 && hPad < height &&
              wPad >= 0 && wPad < width) {
              volData(((cVol * depth + tPad) * height + hPad) * width + wPad + volDataOffset) +=
                colData(((c * depthCol + t) * heightCol + h) * widthCol + w + colDataOffset)
            }
            w += 1
          }
          h += 1
        }
        t += 1
      }
      c += 1
    }
  }
}
