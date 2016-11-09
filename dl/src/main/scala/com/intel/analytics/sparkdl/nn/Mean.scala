/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.sparkdl.nn

import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * It is a simple layer which applies a mean operation over the given dimension.
 * When nInputDims is provided, the input will be considered as a batches.
 * Then the mean operation will be applied in (dimension + 1)
 * @param dimension the dimension to be applied mean operation
 * @param nInputDims the number of dimensions of the give input
 */
class Mean[T: ClassTag](
  dimension: Int = 1,
  nInputDims: Int = -1)
  (implicit ev: TensorNumeric[T]) extends Sum[T](dimension, nInputDims, true) {
  override def toString: String = s"nn.Mean"
}
