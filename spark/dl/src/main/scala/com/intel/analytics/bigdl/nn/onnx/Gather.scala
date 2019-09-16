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
 * Given data tensor of rank r >= 1, and indices tensor of rank q, gather entries of the
 * axis dimension of data (by default outer-most one as axis=0) indexed by indices, and
 * concatenates them in an output tensor of rank q + (r - 1).
 * Example 1:
 *    data = [ [1.0, 1.2], [2.3, 3.4], [4.5, 5.7], ]
 *    indices = [ [0, 1], [1, 2], ]
 *    output = [ [ [1.0, 1.2], [2.3, 3.4], ], [ [2.3, 3.4], [4.5, 5.7], ], ]
 * Example 2:
 *    data = [ [1.0, 1.2, 1.9], [2.3, 3.4, 3.9], [4.5, 5.7, 5.9], ]
 *    indices = [ [0, 2], ] axis = 1,
 *    output = [ [ [1.0, 1.9], [2.3, 3.9], [4.5, 5.9], ], ]
 */
object Gather {
  def apply[T: ClassTag, D: ClassTag](
    axis: Int = 0
  )(implicit ev: TensorNumeric[T], ev2: TensorNumeric[D]):
  nn.ops.Gather[T, D] = new nn.ops.Gather()
}
