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
import com.intel.analytics.bigdl.tensor.TensorNumericMath._
import com.intel.analytics.bigdl.utils.{Engine, Table}

import scala.concurrent.Future
import scala.reflect.ClassTag

/**
 * This operation creates a new tensor by replicating input multiples times.
 * The output tensor's i'th dimension has input.dims(i) * multiples[i] elements,
 * and the values of input are replicated multiples[i] times along the 'i'th dimension.
 *
 * For example, tiling [a b c d] by [1, 2] produces [a b c d a b c d].
 *
 * @param ev$1
 * @param ev
 * @tparam T Numeric type. Only support float/double now
 */
class Tile[T: ClassTag]()(implicit ev: TensorNumeric[T])
  extends Operation[Table, Tensor[_], T] {
  @transient
  private var results: Array[Future[Unit]] = _

  def updateOutput(inputs: Table): Tensor[_] = {
    val input = inputs[Tensor[Tensor[NumericWildcard]]](1)
    val multiples = inputs[Tensor[Int]](2)

    if (multiples.isEmpty) {
      output = input
      return output
    }

    require(input.nDimension() == multiples.size(1),
      "Length of multiples must be the same as the number of dimensions in input")

    output.asInstanceOf[Tensor[Tensor[NumericWildcard]]].resizeAs(input).copy(input)

    for (j <- 1 to input.nDimension()) {
      val currentOutput = output.clone()
      val mult = multiples(Array(j))
      val newSize = output.size()
      newSize(j - 1) = newSize(j - 1) * mult
      output.resize(newSize)
      var offset = 1
      var i = 0
      while (i < mult) {
        val _offset = offset

        if (results == null || results.length != mult) {
          results = new Array[Future[Unit]](mult)
        }

        results(i) = Engine.model.invoke(() => {
          val target = this.output.narrow(j, _offset,
            currentOutput.size(j))
          if (target.isContiguous() || j > 2) {
            // Copy directly when target is Contiguous or dimension is larger than 2
            // in which case the contiguous region in target tensor is fairly small in practice
            target.asInstanceOf[Tensor[NumericWildcard]]
              .copy(currentOutput.asInstanceOf[Tensor[NumericWildcard]])
          } else {
            // Divide target into contiguous frames when target isn't contiguous
            var f = 1
            while (f <= target.size(1)) {
              val curFrame = target.select(1, f)
              val outputFrame = currentOutput.select(1, f)
              require(curFrame.isContiguous())
              require(outputFrame.isContiguous())
              curFrame.asInstanceOf[Tensor[NumericWildcard]]
                .copy(outputFrame.asInstanceOf[Tensor[NumericWildcard]])
              f += 1
            }
          }
        })
        i += 1
        offset += currentOutput.size(j)
      }
    }

    output
  }
}

object Tile {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): Operation[Activity, Activity, T]
  = ModuleToOperation[T](new Tile())
}
