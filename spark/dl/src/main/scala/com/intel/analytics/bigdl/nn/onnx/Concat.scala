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

package com.intel.analytics.bigdl.nn.onnx

import scala.reflect.ClassTag
import com.intel.analytics.bigdl.nn
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric


/**
 * Concatenate a list of tensors into a single tensor
 */
object Concat {
  def apply[T: ClassTag](
        nInputDims: Int, // specify the number of dimensions of input, BigDL requires.
        axis: Int = 0 // to be join in this dimension
    )(implicit ev: TensorNumeric[T]): nn.JoinTable[T] = {

    // Todo: investigate attribute nInputDims
    // It seems it doesn't take N or C as a dimension, if input is in the form of (N, C, W, H).
    // So a Tensor with (n, c, w, h), JoinTable regards nInputDims is 3 instead of 4,
    // otherwise would get an IndexOutofBound exceaption
    new nn.JoinTable(dimension = axis + 1, nInputDims = nInputDims)
  }
}
