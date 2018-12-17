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

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class Pad[T: ClassTag, D: ClassTag](
  val mode: String,
  val constantValue: Double)
  (implicit ev: TensorNumeric[T]) extends Operation[Table, Tensor[D], T] {
  output = Activity.allocate[Tensor[D], D]()

  def constantPadding(
    input: Tensor[D],
    padding: Tensor[Int],
    output: Tensor[D]
  ): Unit = {
    val inputOffset = input.storageOffset() - 1
    val inputData = input.storage()
    val outputOffset = output.storageOffset() - 1
    val outputData = output.storage()

    val dim = input.nDimension()
    val sizes = input.size()

    var dim6Flag = true
    var dim5Flag = true
    var dim4Flag = true
    var dim3Flag = true
    var dim2Flag = true
    var dim1Flag = true

    var dim6OuputOffset = 0
    var dim5OuputOffset = 0
    var dim4OuputOffset = 0
    var dim3OuputOffset = 0
    var dim2OuputOffset = 0
    var dim1OuputOffset = 0

    var dim6InputOffset = 0
    var dim5InputOffset = 0
    var dim4InputOffset = 0
    var dim3InputOffset = 0
    var dim2InputOffset = 0
    var dim1InputOffset = 0

    var i = 0
    while (dim6Flag && i < (if (dim - 6 >= 0) sizes(dim - 6) else Integer.MAX_VALUE)) {
      if (dim - 6 < 0) {
        dim6Flag = false
      } else {
        dim6OuputOffset = (i + padding(Array(dim - 5, 1))) * output.size(dim) *
          output.size(dim - 1) * output.size(dim - 2) * output.size(dim - 3) * output.size(dim - 4)
        dim6InputOffset =
          i * input.size(dim) * input.size(dim - 1) * input.size(dim - 2) *
            input.size(dim - 3) * input.size(dim - 4)
      }
      var j = 0
      while (dim5Flag && j < (if (dim - 5 >= 0) sizes(dim - 5) else Integer.MAX_VALUE)) {
        if (dim - 5 < 0) {
          dim5Flag = false
        } else {
          dim5OuputOffset = (j + padding(Array(dim - 4, 1))) *
            output.size(dim) * output.size(dim - 1) * output.size(dim - 2) * output.size(dim - 3)
          dim5InputOffset = j * input.size(dim) * input.size(dim - 1) *
            input.size(dim - 2) * input.size(4)
        }
        var k = 0
        while (dim4Flag && k < (if (dim - 4 >= 0) sizes(dim - 4) else Integer.MAX_VALUE)) {
          if (dim - 4 < 0) {
            dim4Flag = false
          } else {
            dim4OuputOffset = (k + padding(Array(dim - 3, 1))) *
              output.size(dim) * output.size(dim - 1) * output.size(dim - 2)
            dim4InputOffset = k * input.size(dim) * input.size(dim - 1) * input.size(dim - 2)
          }
          var l = 0
          while (dim3Flag && l < (if (dim - 3 >= 0) sizes(dim - 3) else Integer.MAX_VALUE)) {
            if (dim - 3 < 0) {
              dim3Flag = false
            } else {
              dim3OuputOffset = (l + padding(Array(dim - 2, 1))) *
                output.size(dim) * output.size(dim - 1)
              dim3InputOffset = l * input.size(dim) * input.size(dim - 1)
            }
            var m = 0
            while (dim2Flag && m < (if (dim - 2 >= 0) sizes(dim - 2) else Integer.MAX_VALUE)) {
              if (dim - 2 < 0) {
                dim2Flag = false
              } else {
                dim2OuputOffset = (m + padding(Array(dim - 1, 1))) * output.size(dim)
                dim2InputOffset = m * input.size(dim)
              }
              var n = 0
              while (dim1Flag && n < (if (dim - 1 >= 0) sizes(dim - 1) else Integer.MAX_VALUE)) {
                if (dim - 1 < 0) {
                  dim1Flag = false
                } else {
                  dim1OuputOffset = n + padding(Array(dim, 1))
                  dim1InputOffset = n
                }

                outputData(outputOffset + dim6OuputOffset + dim5OuputOffset
                  + dim4OuputOffset + dim3OuputOffset + dim2OuputOffset +
                  dim1OuputOffset) =
                  inputData(inputOffset + dim6InputOffset + dim5InputOffset
                    + dim4InputOffset + dim3InputOffset + dim2InputOffset +
                    dim1InputOffset)
                n += 1
              }
              m += 1
            }
            l += 1
          }
          k += 1
        }
        j += 1
      }
      i += 1
    }
  }

  def updateOutput(inputs: Table): Tensor[D] = {
    val input = inputs[Tensor[D]](1)
    val padding = inputs[Tensor[Int]](2)

    require(padding.size() sameElements Array(input.nDimension(), 2),
      "the padding tensor must be an integer tensor with shape [n, 2]," +
        "where n is the number of dimension of input")

    val resize = new Array[Int](input.nDimension())
    for (i <- 1 to input.nDimension()) {
      resize(i - 1) = input.size(i) + padding(Array(i, 1)) + padding(Array(i, 2))
    }
    output.resize(resize)

    mode match {
      case "CONSTANT" => constantPadding(input, padding, output)
    }

    output
  }

  override def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev))
  }
}

object Pad {
  def apply[T: ClassTag, D: ClassTag](
    mode: String,
    constantValue: Double)
    (implicit ev: TensorNumeric[T]): Operation[Activity, Activity, T]
  = ModuleToOperation[T](
    new Pad[T, D](mode = mode, constantValue = constantValue))
}
