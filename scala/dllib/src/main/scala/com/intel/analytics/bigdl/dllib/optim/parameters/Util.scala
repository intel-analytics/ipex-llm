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

package com.intel.analytics.bigdl.parameters

import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

private[bigdl] object Util {
  /** Get square sum of a tensor in parallel, which has better
   * performance if tensor is in large size
   * @param parameters
   * @param parallelism
   * @return square sum of the tensor
   */
  def getSumsquareInParallel[T](parameters: Tensor[T], parallelism: Int)
                               (implicit ev: TensorNumeric[T]): Double = {
    val gradLength = parameters.nElement()
    val taskSize = gradLength / parallelism
    val extraTask = gradLength % parallelism
    val parallelNum = if (taskSize == 0) extraTask else parallelism
    val squares = new Array[Double](parallelNum)
    Engine.default.invokeAndWait((0 until parallelNum).map(tid => () => {
      val offset = tid * taskSize + math.min(tid, extraTask)
      val length = taskSize + (if (tid < extraTask) 1 else 0)
      squares(tid) = ev.toType[Double](
        parameters.narrow(1, offset + 1, length).sumSquare())
    }))
    var sum = 0.0
    var i = 0
    while (i < parallelNum) {
      sum += squares(i)
      i += 1
    }
    sum
  }
}
