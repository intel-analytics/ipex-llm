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

package com.intel.analytics.bigdl.models.mnist

import com.intel.analytics.bigdl.nn.{Linear, LogSoftMax, SpatialMaxPooling, _}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

object LeNet5 {
  def apply[T: ClassTag](classNum: Int)
    (implicit ev: TensorNumeric[T]): Module[Tensor[T], Tensor[T], T] = {
    val model = Sequential[Tensor[T], Tensor[T], T]()
    model.add(Reshape[T](Array(1, 28, 28)))
    model.add(SpatialConvolution[T](1, 6, 5, 5))
    model.add(Tanh[T]())
    model.add(SpatialMaxPooling[T](2, 2, 2, 2))
    model.add(Tanh[T]())
    model.add(SpatialConvolution[T](6, 12, 5, 5))
    model.add(SpatialMaxPooling[T](2, 2, 2, 2))
    model.add(Reshape[T](Array(12 * 4 * 4)))
    model.add(Linear[T](12 * 4 * 4, 100))
    model.add(Tanh[T]())
    model.add(Linear[T](100, classNum))
    model.add(LogSoftMax[T]())
    model
  }
}
