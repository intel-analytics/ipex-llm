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

package com.intel.analytics.sparkdl.models.cifar

import com.intel.analytics.sparkdl.nn.{Linear, LogSoftMax, Reshape, _}
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

object SimpleCNN {

  def apply[T: ClassTag](classNum: Int)(implicit ev: TensorNumeric[T]): Module[T] = {
    val model = new Sequential[T]

    model.add(new SpatialConvolutionMap[T](SpatialConvolutionMap.random[T](3, 16, 1), 5, 5))
    model.add(new Tanh[T]())
    model.add(new SpatialMaxPooling[T](2, 2, 2, 2))
    /* stage 2 : filter bank -> squashing -> max pooling */
    model.add(new SpatialConvolutionMap[T](SpatialConvolutionMap.random[T](16, 256, 4), 5, 5))
    model.add(new Tanh[T]())
    model.add(new SpatialMaxPooling[T](2, 2, 2, 2))
    /* stage 3 : standard 2-layer neural network */
    model.add(new Reshape[T](Array(256 * 5 * 5)))
    model.add(new Linear[T](256 * 5 * 5, 128))
    model.add(new Tanh[T]())
    model.add(new Linear[T](128, classNum))
    model.add(new LogSoftMax[T]())

    model
  }
}
