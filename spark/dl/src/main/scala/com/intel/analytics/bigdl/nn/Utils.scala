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

import com.google.protobuf.ByteString
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, DataFormat}
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.collection.mutable
import scala.reflect.ClassTag

object Utils {
  /**
   * This method recursively keep the shape of the source table `t2` and
   * set the elements of each tensor to zero, saving the result on the destination
   * table `t1`
   * Notice that `t1` and `t2` can only contain tables or tensors
   * @param t1 is the destination table
   * @param t2 is the source table
   * @return
   */
  def zeroTableCopy[T : ClassTag](t1: Table, t2: Table)(
    implicit ev: TensorNumeric[T]): Table = {
    t2.foreach { case ((k: Any, v: Any)) =>
      if (v.isInstanceOf[Table]) {
        t1.update(k, zeroTableCopy(if (t1.contains(k)) t1(k) else T(), t2(k)))
      } else {
        require(v.isInstanceOf[Tensor[_]], "Input can only consist of Tensor or Table")
        val tensorV = v.asInstanceOf[Tensor[T]]
        if (!t1.contains(k)) {
          t1.update(k, tensorV.clone().zero())
        } else {
          t1[Tensor[T]](k).resizeAs(tensorV)
          t1[Tensor[T]](k).zero()
        }
      }
    }
    t1.foreach { case ((k: Any, v: Any)) =>
      if (!t2.contains(k)) {
        t1.update(k, null)
      }
    }

    t1
  }

  /**
   * Resize table target as table src.
   *
   * @param target
   * @param src
   */
  def recursiveResizeAs[T : ClassTag](target : Activity, src: Activity)(
    implicit ev: TensorNumeric[T]): Activity = {
    var result: Activity = null
    if (src.isInstanceOf[Table]) {
      val srcTable = src.toTable
      result = if (null == target) {
        T()
      } else if (target.isInstanceOf[Tensor[_]]) {
        T(target)
      } else {
        target
      }

      val resultTable = result.toTable
      var i = 1
      while (i <= src.toTable.length()) {
        if (resultTable.contains(i)) {
          resultTable(i) = recursiveResizeAs(resultTable(i), srcTable(i))
        } else {
          resultTable(i) = recursiveResizeAs(null, srcTable(i))
        }
        i += 1
      }
      while (i <= resultTable.length()) {
        resultTable.remove(i)
        i += 1
      }
    } else if (src.isInstanceOf[Tensor[_]]) {
      result = if (target.isInstanceOf[Tensor[_]]) {
        target
      } else {
        Tensor[T]()
      }
      result.toTensor[T].resizeAs(src.toTensor)
    }
    result
  }

  /**
   * Apply function 'func' on all tensor in the table.
   *
   * @param x
   * @param func
   */
  def recursiveTensorApply1[T](x: Activity, func: Tensor[T] => Tensor[T])(
    implicit ev: TensorNumeric[T]): Unit = {
    require(x.isInstanceOf[Activity],
      s"expecting tensors or tables thereof. Got ${x} instead"
    )
    if (x.isInstanceOf[Table]) {
      var i = 1
      while (i <= x.toTable.length()) {
        recursiveTensorApply1(x.toTable(i), func)
        i += 1
      }
    } else {
      func(x.toTensor[T])
    }
  }

  /**
   * Apply function 'func' on each tensor in table x and table y recursively.
   *
   * Table x should have the same size with table y.
   *
   * @param x
   * @param y
   * @param func
   * @return
   */
  def recursiveTensorApply2[T](x: Activity, y: Activity,
    func: (Tensor[T], Tensor[T]) => Tensor[T])(implicit ev: TensorNumeric[T]): Activity = {
    if (y.isInstanceOf[Tensor[_]] && x.isInstanceOf[Tensor[_]]) {
      require(x.toTensor[T].nElement() == y.toTensor[T].nElement(),
        "x, y should have the same size" +
          s"x size ${x.toTensor[T].nElement()}, y size ${y.toTensor[T].nElement()}")
      func(x.toTensor[T], y.toTensor[T])
    } else {
      require(x.isInstanceOf[Table] && y.isInstanceOf[Table], "x, y should have the same size")
      require(x.toTable.length() == y.toTable.length(), "x, y should have the same size" +
        s"x size ${x.toTable.length()}, y size ${y.toTable.length()}")
      var i = 1
      while (i <= x.toTable.length()) {
        recursiveTensorApply2[T](x.toTable(i), y.toTable(i), func)
        i += 1
      }
    }
    x
  }

  /**
   * Apply a add operation on table x and table y one by one.
   * y := y + alpha * x
   *
   * Table x should have the same size with y.
   *
   * @param y
   * @param alpha
   * @param x
   * @tparam T: Float or Double
   * @return y
   */
  def recursiveAdd[T](y: Activity, alpha: Double = 1.0, x: Activity )(
    implicit ev: TensorNumeric[T]): Activity = {
    recursiveTensorApply2[T](y, x, (t1, t2) => t1.add(ev.fromType[Double](alpha), t2))
    y
  }

  /**
   * copy table x's tensor to table y.
   *
   * Table x should have the same size with y.
   *
   * @param y
   * @param x
   * @tparam T: Float or Double
   * @return y
   */
  def recursiveCopy[T](y: Activity, x: Activity )(
    implicit ev: TensorNumeric[T]): Activity = {
    recursiveTensorApply2[T](y, x, (t1, t2) => t1.copy(t2))
    y
  }

  /**
   * Fill the value to each Tensor in the table recursively
   *
   * @param x
   * @param value
   */
  def recursiveFill[T](x: Activity, value : Double)(
    implicit ev: TensorNumeric[T]): Unit = {
    recursiveTensorApply1[T](x, t => t.fill(ev.fromType[Double](value)))
  }

  /**
   * get all modules and map by name
   *
   * @param model
   * @tparam T
   * @return
   */
  def getNamedModules[T](model: Module[T]): Map[String, Module[T]] = {
    var namedModules: Map[String, Module[T]] = Map()
    def getModules(module: Module[T]): Unit = {
      module match {
        case m: Container[_, _, T] =>
          namedModules += (module.getName() -> module)
          for (m <- module.asInstanceOf[Container[_, _, T]].modules) getModules(m)
        case _ => namedModules += (module.getName() -> module)
      }
    }
    getModules(model)
    namedModules
  }

  /**
   * copy src's parameters and running status to dst
   * @param src source model
   * @param dst destination model
   */
  def copyModule[T](src: Module[T], dst: Module[T]): Module[T] = {
    // copy parameters
    val srcParameters = src.getParameters()._1
    val dstParameters = dst.getParameters()._1
    require(srcParameters.size(1) == dstParameters.size(1),
      s"$src and $dst is not the same type.")
    dstParameters.copy(srcParameters)
    // copy running status
    dst.setExtraParameter(src.getExtraParameter())
    dst
  }

  /**
   * get whether the module is layerwise scaled
   * @param model input module
   * @return whether the module is layerwise scaled
   */
  def isLayerwiseScaled[T](model: Module[T]): Boolean = model match {
    case m: Container[Activity, Activity, T] =>
      var i = 0
      while (i < m.modules.length) {
        if (isLayerwiseScaled(m.modules(i))) return true
        i += 1
      }
      false
    case m: Module[T] => (m.getScaleB() != 1) || (m.getScaleW() != 1)
  }

  /**
   * get the inner loop size and outer loop size given a pivot dim
   * @param pivotDim is the dim whose value larger than 1
   * @return inner loop size and outer loop size
   */
  private[nn] def getInnerOuterNum[T](pivotDim: Int, data: Tensor[T]): (Int, Int) = {
    var k = 1
    var outerNum = 1
    while (k < pivotDim) {
      outerNum *= data.size(k)
      k += 1
    }
    var innerNum = 1
    k = pivotDim + 1
    while (k <= data.dim()) {
      innerNum *= data.size(k)
      k += 1
    }
    (innerNum, outerNum)
  }

  /**
   * if there is only one dim of size > 1, return this dim(count from 1)
   * else return -1
   * e.g. (1, 2, 1, 1) returns 1, (1, 2, 3, 1) returns -1, and (1, 1, 1, 1) returns -1
   * @param size size of tensor
   * @return (the only dim whose value > 1) else (-1)
   */
  private[nn] def getOnlyDimGtOne(size: Array[Int]): Int = {
    var i = 0
    var count = 0
    var pivot = 0
    while (i < size.length) {
      if (size(i) > 1) {
        count += 1
        pivot = i + 1
      }
      i += 1
    }
    if (count == 1) pivot else -1
  }

  /**
   *
   * @return Array(padTop, padBottom, padLeft, padRight, outputHeight, outputWidth)
   *         or Array(padFront, padBackward, padTop, padBottom, padLeft, padRight,
   *         outputDepth, outputHeight, outputWidth)
   */
  private[nn] def getSAMEOutSizeAndPadding(
    inputHeight: Int,
    inputWidth: Int,
    dH: Int,
    dW: Int,
    kH: Int,
    kW: Int,
    inputDepth: Int = -1,
    dT: Int = -1,
    kT: Int = -1): Array[Int] = {
    val oW = Math.ceil(inputWidth.toFloat / dW.toFloat).toInt
    val oH = Math.ceil(inputHeight.toFloat / dH.toFloat).toInt
    val padAlongWidth = Math.max(0, (oW -1) * dW + kW - inputWidth)
    val padAlongHeight = Math.max(0, (oH - 1) * dH + kH - inputHeight)
    if (inputDepth != -1) {
      require(dT > 0 && kT > 0, "kernel size and strideSize cannot be smaller than 0")
      val oT = Math.ceil(inputDepth.toFloat / dT.toFloat).toInt
      val padAlongDepth = Math.max(0, (oT -1) * dT + kT - inputDepth)
      return Array(padAlongDepth/2, padAlongDepth - padAlongDepth/2, padAlongHeight/2,
        padAlongHeight - padAlongHeight/2, padAlongWidth/2, padAlongWidth - padAlongWidth/2,
        oT, oH, oW)
    }
    Array(padAlongHeight/2, padAlongHeight - padAlongHeight/2,
      padAlongWidth/2, padAlongWidth - padAlongWidth/2,
        oH, oW)
  }

  /**
   *
   * @return Array(padLeft, padRight, padTop, padBottom, outputHeight, outputWidth)
   *         or Array(padFront, padBack, padLeft, padRight, padTop, padBottom,
   *         outputDepth, outputHeight, outputWidth)
   */
  private[nn] def getOutSizeAndPadding(
    inputHeight: Int,
    inputWidth: Int,
    dH: Int,
    dW: Int,
    kH: Int,
    kW: Int,
    padH: Int,
    padW: Int,
    ceilMode: Boolean,
    dilationHeight: Int = 1,
    dilationWidth: Int = 1,
    inputdepth: Int = -1,
    dt: Int = -1,
    kt: Int = -1,
    padt: Int = 0,
    dilationDepth: Int = 1): Array[Int] = {
    var oheight = 0
    var owidth = 0
    var odepth = 0

    val dilationKernelHeight = dilationHeight * (kH - 1) + 1
    val dilationKernelWidth = dilationWidth * (kW - 1) + 1
    val dilationKernelDepth = if (inputdepth > 0) dilationDepth * (kt - 1) + 1 else kt

    if (ceilMode) {
      oheight = math.ceil(1.0 * (inputHeight - dilationKernelHeight + 2*padH) / dH).toInt + 1
      owidth = math.ceil(1.0 * (inputWidth - dilationKernelWidth + 2*padW) / dW).toInt + 1
      if (inputdepth > 0) {
        require(dt > 0 && kt > 0 && padt >= 0,
          "kernel size, stride size, padding size cannot be smaller than 0")
        odepth = math.ceil(1.0 * (inputdepth - dilationKernelDepth + 2*padt) / dt).toInt + 1
      }
    } else {
      oheight = math.floor(1.0 * (inputHeight - dilationKernelHeight + 2*padH) / dH).toInt + 1
      owidth = math.floor(1.0 * (inputWidth - dilationKernelWidth + 2*padW) / dW).toInt + 1
      if (inputdepth > 0) {
        require(dt > 0 && kt > 0 && padt >= 0,
          "kernel size, stride size, padding size cannot be smaller than 0")
        odepth = math.floor(1.0 * (inputdepth - dilationKernelDepth + 2*padt) / dt).toInt + 1
      }
    }

    if (padH != 0 || padW != 0 || padt != 0) {
      if ((oheight - 1) * dH >= inputHeight + padH) oheight -= 1
      if ((owidth - 1) * dW >= inputWidth + padW) owidth -= 1
      if (inputdepth > 0) {
        if ((odepth - 1) * dt >= inputdepth + padt) odepth -= 1
        return Array(padt, padt, padH, padH, padW, padW, odepth, oheight, owidth)
      }
    } else if (inputdepth > 0) {
        return Array(padt, padt, padH, padH, padW, padW, odepth, oheight, owidth)
    }
    Array(padH, padH, padW, padW, oheight, owidth)
  }

  private[nn] def getOutSizeAndPaddingForDNN(
    inputHeight: Int,
    inputWidth: Int,
    dH: Int,
    dW: Int,
    kH: Int,
    kW: Int,
    padH: Int,
    padW: Int,
    ceilMode: Boolean,
    dilationHeight: Int = 1,
    dilationWidth: Int = 1,
    inputdepth: Int = -1,
    dt: Int = -1,
    kt: Int = -1,
    padt: Int = 0,
    dilationDepth: Int = 1): Array[Int] = {
    // compute padding left, right, top and bottom
    var pad_t = padH
    var pad_b = padH
    var pad_l = padW
    var pad_r = padW

    var oheight = 0
    var owidth = 0
    var odepth = 0

    val dilationKernelHeight = dilationHeight * (kH - 1) + 1
    val dilationKernelWidth = dilationWidth * (kW - 1) + 1

    oheight = math.ceil(1.0 * (inputHeight - dilationKernelHeight + 2*padH) / dH).toInt + 1
    owidth = math.ceil(1.0 * (inputWidth - dilationKernelWidth + 2*padW) / dW).toInt + 1

    if (padH != 0 || padW != 0 || padt != 0 || kH == 1 || kW == 1) {
      if ((oheight - 1) * dH >= inputHeight + padH) oheight -= 1
      if ((owidth - 1) * dW >= inputWidth + padW) owidth -= 1
    }

    val h = inputHeight + pad_t
//    var pad_b = padH
    while ((h + pad_b) < (dH * (oheight - 1) + kH)) pad_b = pad_b + 1
    val w = inputWidth + pad_l
//    var pad_r = padW
    while ((w + pad_r) < (dW * (owidth - 1) + kW)) pad_r = pad_r + 1

    Array(pad_t, pad_b, pad_l, pad_r, oheight, owidth)
  }

  private[nn] def getOutputShape(outputHeight: Int, outputWidth: Int, nOutputPlane: Int,
    batchSize: Int = -1, format: DataFormat): Array[Int] = {
    format match {
      case DataFormat.NCHW =>
        if (batchSize == -1) {
          Array(nOutputPlane, outputHeight, outputWidth)
        } else {
          Array(batchSize, nOutputPlane, outputHeight, outputWidth)
        }
      case DataFormat.NHWC =>
        if (batchSize == -1) {
          Array(outputHeight, outputWidth, nOutputPlane)
        } else {
          Array(batchSize, outputHeight, outputWidth, nOutputPlane)
        }

    }
  }

  private[nn] def getOutputSize(inputSize: Int, filterSize: Int,
                    stride: Int, padding: String) = {
    padding.toLowerCase() match {
      case "valid" =>
        val outputSize = (inputSize - filterSize + stride) / stride
        (outputSize, 0, 0)
      case "same" =>
        val outputSize = (inputSize + stride - 1) / stride
        val paddingNeeded = math.max(0, (outputSize - 1) * stride + filterSize - inputSize)
        val padBefore = paddingNeeded / 2
        val padAfter = paddingNeeded - padBefore
        (outputSize, padBefore, padAfter)
    }
  }

  def shuffle[T: ClassTag](src: Tensor[T], permutation: Array[Int], buffer: Tensor[T] = null)(
    implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(permutation.length == src.nDimension,
      s"permutation length should be same as tensor dimension")
    require(permutation.min >= 0 && permutation.max <= src.size().max,
      s"permutation min value should be between 0 and ${src.size().max}")
    require(permutation.distinct.size == src.nDimension, s"permutation has duplicated input")

    var i = 0
    val outSize = new Array[Int](src.nDimension)
    while (i < permutation.length) {
      outSize(i) = src.size(permutation(i))
      i += 1
    }

    val out = if (buffer == null) {
      Tensor[T]()
    } else {
      buffer
    }

    out.resize(outSize)

    i = 0
    val numOfElements = src.nElement()
    while (i < numOfElements) {
      var srcIndex = 0
      var tmp = i

      var j = 1
      while (j <= src.nDimension) {
        val curDim = tmp / out.stride(j)
        tmp %= out.stride(j)

        srcIndex += curDim * src.stride(permutation(j - 1))

        j += 1
      }

      out.storage().array()(i) = src.storage().array()(srcIndex)
      i += 1
    }

    out
  }

  private[nn] def getPaddingAndOutputSize(
    inputHeight: Int,
    inputWidth: Int,
    dH: Int,
    dW: Int,
    kH: Int,
    kW: Int,
    padH: Int,
    padW: Int,
    ceilMode: Boolean = false
  ): (Int, Int, Int, Int, Int, Int) = {
    // compute padding left, right, top and bottom
    var pad_t = padH
    var pad_b = padH
    var pad_l = padW
    var pad_r = padW

    var oheight = 0
    var owidth = 0
    var odepth = 0

    if (ceilMode) {
      oheight = math.ceil(1.0 * (inputHeight - kH + 2 * padH) / dH).toInt + 1
      owidth = math.ceil(1.0 * (inputWidth - kW + 2 * padW) / dW).toInt + 1
    } else {
      oheight = math.floor(1.0 * (inputHeight - kH + 2 * padH) / dH).toInt + 1
      owidth = math.floor(1.0 * (inputWidth - kW + 2 * padW) / dW).toInt + 1
    }
    if (padH != 0 || padW != 0 || kH == 1 || kW == 1) {
      if ((oheight - 1) * dH >= inputHeight + padH) oheight -= 1
      if ((owidth - 1) * dW >= inputWidth + padW) owidth -= 1
    }

    val h = inputHeight + pad_t
    while ((h + pad_b) < (dH * (oheight - 1) + kH)) pad_b = pad_b + 1
    val w = inputWidth + pad_l
    while ((w + pad_r) < (dW * (owidth - 1) + kW)) pad_r = pad_r + 1

    (pad_t, pad_b, pad_l, pad_r, oheight, owidth)
  }
  /**
   * Calculate forward time and backward time.
   * @param times
   * @tparam T
   * @return
   */
  def calculateFwdBwdTime[T: ClassTag](
    times: Array[(AbstractModule[_ <: Activity, _ <: Activity, T], Long, Long)]): (Long, Long) = {
      times.map(t => (t._2, t._3)).reduce((a, b) => (a._1 + b._1, a._2 + b._2))
  }

  /**
   * calculate scales of tensor based on the mask
   *
   * The mask parameter determines the dimension to which the scales array is applied to.
   * If the ith bit of mask is set, it will select that dimension and calc scales on that.
   * For a 5-dimensional tensor T[g0, o1,i2,h3,w4] where the numbering indicates the bit-index:
   *    + A mask = 3 = $2^0 | 2^1$ selects the group (g0) and output channels (o1).
   *    + A mask = 2 = $2^1$ selects the output channels (o1).
   * For a [4, 3, 2, 2] tensor and 3 ( $2^0|2^1$ ) as the mask, it will generate 4*3=12 max values.
   *
   * @param tensor the tensor want to be caculated
   * @param mask the mask value. You can construct it with math.pow(2, ?).
   * @return the scales of tensor relevant with mask
   */
  private[nn] def calcScales(tensor: Tensor[Float], mask: Int): Array[Float] = {
    // inner helper function
    def calcScalesHelper(tensor: Tensor[Float], maskStr: String,
      result: mutable.ListBuffer[Float], index: Int): Unit = {
      if (index < maskStr.length) {
        if (maskStr(index).asDigit == 1) { // mask bit is ON at this dimension
          (1 to tensor.size(index + 1)).foreach(
            i => { // split the tensor based on its size
              calcScalesHelper(tensor.narrow(index + 1, i, 1), maskStr, result, index + 1)
            }
          )
        } else {
          calcScalesHelper(tensor, maskStr, result, index + 1)
        }

      } else { // finished splitting tensor based on its mask bit, aggregate and append the result
        result.append(tensor.clone().abs().max())
      }

    }

    def maskInterval: String = {
      val start = 0
      val end = (math.pow(2, tensor.size().length) - 1).toInt

      s"mask should between [$start, $end]"
    }
    require(mask.toBinaryString.length <= tensor.size().length, s"$maskInterval")

    val result = mutable.ListBuffer[Float]()

    calcScalesHelper(tensor, mask.toBinaryString.reverse, result, 0 /* start dimension */)

    result.toArray
  }
}
