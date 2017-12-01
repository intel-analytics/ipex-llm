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

import com.intel.analytics.bigdl.nn.SpatialConvolution
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class DepthwiseConv2DNative[T: ClassTag](
  sW: Int, sH: Int,
  pW: Int, pH: Int,
  dataFormat: DataFormat
)(implicit ev: TensorNumeric[T])
extends Operation[Table, Tensor[T], T] {
  override def updateOutput(input: Table): Tensor[T] = {
    require(input.length() == 2, "Input must contain 2 inputs")
    val data = input[Tensor[T]](1)
    val filter = input[Tensor[T]](2)
  }
}
