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
package com.intel.analytics.bigdl.nn.ops

import com.intel.analytics.bigdl.nn.Utils
import com.intel.analytics.bigdl.tensor.{DoubleType, FloatType, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class Dilation2DBackpropInput[T: ClassTag, D: ClassTag](strides: Array[Int],
                                                        rates: Array[Int],
                                                        padding: String)
         (implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends Operation[Table, Tensor[D], T]{

  output = Tensor[D]

  private def dilationBackpropInputFloat(input: Tensor[Float],
                                    filter: Tensor[Float],
                                    outBackprop: Tensor[Float],
                                    inputBackprop: Tensor[Float],
                                    strideRows: Int, strideCols: Int,
                                    rateRows: Int, rateCols: Int) = {

    val batch = input.size(1)
    val inputRows = input.size(2)
    val inputCols = input.size(3)
    val depth = input.size(4)

    val filterRows = filter.size(1)
    val filterCols = filter.size(2)

    val filterRowsEff = filterRows + (filterRows - 1) * (rateRows - 1)
    val filterColsEff = filterCols + (filterCols - 1) * (rateCols - 1)

    val (outputRows, padTop, _) =
      Utils.getOutputSize(inputRows, filterRowsEff, strideRows, padding)
    val (outputCols, padLeft, _) =
      Utils.getOutputSize(inputCols, filterColsEff, strideCols, padding)

    inputBackprop.resizeAs(input)

    val inputData = input.storage().array()
    val inputDataOffset = input.storageOffset() - 1

    val filterData = filter.storage().array()
    val filterDataOffset = filter.storageOffset() - 1

    val outBackpropData = outBackprop.storage().array()
    val outBackpropDataOffset = outBackprop.storageOffset() - 1

    val inputBackpropData = inputBackprop.storage().array()
    val inputBackpropDataOffset = inputBackprop.storageOffset() - 1

    var b = 0
    while (b < batch) {
      var h_out = 0
      while (h_out < outputRows) {
        val h_beg = h_out * strideRows - padTop
        var w_out = 0
        while (w_out < outputCols) {
          val w_beg = w_out * strideCols - padLeft
          var d = 0
          while (d < depth) {
            var cur_val = Float.MinValue
            var h_in_max = if (h_beg < 0) 0 else h_beg
            var w_in_max = if (w_beg < 0) 0 else w_beg
            var h = 0
            while (h < filterRows) {
              val h_in = h_beg + h * rateRows
              if (h_in >= 0 && h_in < inputRows) {
                var w = 0
                while (w < filterCols) {
                  val w_in = w_beg + w * rateCols
                  if (w_in >= 0 && w_in < inputCols) {
                    val inputIndex = ((b * inputRows + h_in) * inputCols + w_in) * depth + d
                    val inputValue = inputData(inputDataOffset + inputIndex)
                    val filterIndex = (h * filterCols + w) * depth + d
                    val filterValue = filterData(filterDataOffset + filterIndex)
                    val value: Float = inputValue + filterValue
                    if (value > cur_val) {
                      cur_val = value
                      h_in_max = h_in
                      w_in_max = w_in
                    }
                  }
                  w += 1
                }
              }
              h += 1
            }
            val inputBackPropIndex =
              ((b * inputRows + h_in_max) * inputCols + w_in_max) * depth + d
            val outputBackPropIndex =
              ((b * outputRows + h_out) * outputCols + w_out) * depth + d
            inputBackpropData(inputBackpropDataOffset + inputBackPropIndex) +=
              outBackpropData(outBackpropDataOffset + outputBackPropIndex)
            d += 1
          }
          w_out += 1
        }
        h_out += 1
      }
      b += 1
    }

  }



  private def dilationBackpropInputDouble(input: Tensor[Double],
                                    filter: Tensor[Double],
                                    outBackprop: Tensor[Double],
                                    inputBackprop: Tensor[Double],
                                    strideRows: Int, strideCols: Int,
                                    rateRows: Int, rateCols: Int) = {
    val batch = input.size(1)
    val inputRows = input.size(2)
    val inputCols = input.size(3)
    val depth = input.size(4)

    val filterRows = filter.size(1)
    val filterCols = filter.size(2)

    val outputRows = outBackprop.size(2)
    val outputCols = outBackprop.size(3)

    val (padTop, padLeft) = padding.toLowerCase() match {
      case "same" =>
        val top = (outputRows - inputRows) / 2
        val left = (outputCols - inputCols) / 2
        (top, left)
      case "valid" =>
        (0, 0)
    }

    inputBackprop.resizeAs(input)

    val inputData = input.storage().array()
    val inputDataOffset = input.storageOffset() - 1

    val filterData = filter.storage().array()
    val filterDataOffset = filter.storageOffset() - 1

    val outBackpropData = outBackprop.storage().array()
    val outBackpropDataOffset = outBackprop.storageOffset() - 1

    val inputBackpropData = inputBackprop.storage().array()
    val inputBackpropDataOffset = inputBackprop.storageOffset() - 1

    var b = 0
    while (b < batch) {
      var h_out = 0
      while (h_out < outputRows) {
        val h_beg = h_out * strideRows - padTop
        var w_out = 0
        while (w_out < outputCols) {
          val w_beg = w_out * strideCols - padLeft
          var d = 0
          while (d < depth) {
            var cur_val = Double.MinValue
            var h_in_max = if (h_beg < 0) 0 else h_beg
            var w_in_max = if (w_beg < 0) 0 else w_beg
            var h = 0
            while (h < filterRows) {
              val h_in = h_beg + h * rateRows
              if (h_in >= 0 && h_in < inputRows) {
                var w = 0
                while (w < filterCols) {
                  val w_in = w_beg + w * rateCols
                  if (w_in >= 0 && w_in < inputCols) {
                    val inputIndex = ((b * inputRows + h_in) * inputCols + w_in) * depth + d
                    val inputValue = inputData(inputDataOffset + inputIndex)
                    val filterIndex = (h * filterCols + w) * depth + d
                    val filterValue = filterData(filterDataOffset + filterIndex)
                    val value: Double = inputValue + filterValue
                    if (value > cur_val) {
                      cur_val = value
                      h_in_max = h_in
                      w_in_max = w_in
                    }
                  }
                  w += 1
                }
              }
              h += 1
            }
            val inputBackPropIndex =
              ((b * inputRows + h_in_max) * inputCols + w_in_max) * depth + d
            val outputBackPropIndex =
              ((b * outputRows + h_out) * outputCols + w_out) * depth + d
            inputBackpropData(inputBackpropDataOffset + inputBackPropIndex) +=
              outBackpropData(outBackpropDataOffset + outputBackPropIndex)
            d += 1
          }
          w_out += 1
        }
        h_out += 1
      }
      b += 1
    }
  }

  override def updateOutput(inputs: Table): Tensor[D] = {
    val input = inputs[Tensor[D]](1)
    val filter = inputs[Tensor[D]](2)
    val outBackprop = inputs[Tensor[D]](3)

    require(input.dim() == 4, "input must have 4 dims")
    require(filter.dim() == 3, "filter must have 3 dims")


    val strideRows = strides(1)
    val strideCols = strides(2)

    val rateRows = rates(1)
    val rateCols = rates(2)

    if (ev2.getType() == FloatType) {
      val inputTensor = input.asInstanceOf[Tensor[Float]]
      val filterTensor = filter.asInstanceOf[Tensor[Float]]
      val outBackpropTensor = outBackprop.asInstanceOf[Tensor[Float]]
      val outputTensor = output.asInstanceOf[Tensor[Float]]
      dilationBackpropInputFloat(inputTensor, filterTensor, outBackpropTensor, outputTensor,
        strideRows, strideCols, rateRows, rateCols)
    } else if (ev2.getType() == DoubleType) {
      val inputTensor = input.asInstanceOf[Tensor[Double]]
      val filterTensor = filter.asInstanceOf[Tensor[Double]]
      val outBackpropTensor = output.asInstanceOf[Tensor[Double]]
      val outputTensor = output.asInstanceOf[Tensor[Double]]
      dilationBackpropInputDouble(inputTensor, filterTensor, outBackpropTensor, outputTensor,
        strideRows, strideCols, rateRows, rateCols)
    } else {
      throw new IllegalArgumentException(s"does not support datatype ${ev2.getType()}")
    }

    output
  }

  override def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
    (Array(scala.reflect.classTag[T], scala.reflect.classTag[D]), Array(ev, ev2))
  }
}

object Dilation2DBackpropInput {
  def apply[T: ClassTag, D: ClassTag](strides: Array[Int], rates: Array[Int], padding: String)
       (implicit ev: TensorNumeric[T], ev2: TensorNumeric[D]): Dilation2DBackpropInput[T, D] =
    new Dilation2DBackpropInput(strides, rates, padding)
}

