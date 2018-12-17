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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class InTopK[T: ClassTag](k: Int, startFromZero: Boolean = false)(implicit ev: TensorNumeric[T])
  extends Operation[Table, Tensor[Boolean], T]{

  override def updateOutput(input: Table): Tensor[Boolean] = {

    output = Tensor[Boolean]()

    val predictions = input[Tensor[Float]](1)
    val targets = input[Tensor[Int]](2)

    require(predictions.nDimension() == 2, "predictions should be 2D in InTopK")
    require(targets.nDimension() == 1, "targets shoudl be 1D in InTopK")

    val batchSize = targets.size(1)
    output.resizeAs(targets)
    var i = 1
    while(i <= batchSize) {
      var j = 1
      var largerNum = 0
      val d = if (startFromZero) targets.valueAt(i) + 1 else targets.valueAt(i)
      val element = predictions.valueAt(i, d)
      while(j <= predictions.size(2)) {
        if (element < predictions.valueAt(i, j) && j != d) {
          largerNum += 1
        }
        j += 1
      }
      output.setValue(i, largerNum < k)
      i += 1
    }
    output
  }
}

object InTopK {
  def apply[T: ClassTag](k: Int, startFromZero: Boolean = false)(implicit ev: TensorNumeric[T])
  : InTopK[T] = new InTopK[T](k, startFromZero)
}
