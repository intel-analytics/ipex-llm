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
 * Insert single-dimensional entries to the shape of a tensor.
 * Takes one required argument axes, a list of dimensions that will be inserted.
 * Dimension indices in axes are as seen in the output tensor.
 */
object Unsqueeze {
  def apply[@specialized(Float, Double) T: ClassTag](
        axes: List[Int], // List of non-negative integers, indicate the dimensions to be inserted
        numInputDims: Int = Int.MinValue // BigDL requires
  )(implicit ev: TensorNumeric[T]): nn.Unsqueeze[T] = {
    val pos = axes match {
      case List(elem) => elem + 1 // Todo
      case _ => throw new IllegalArgumentException("Bad axes value: " + axes)
    }
     new nn.Unsqueeze[T](pos = pos, numInputDims = numInputDims)
  }
}
