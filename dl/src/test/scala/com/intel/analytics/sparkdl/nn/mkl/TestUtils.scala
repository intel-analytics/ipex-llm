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

package com.intel.analytics.sparkdl.nn.mkl

import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.tensor.Tensor

import scala.reflect.ClassTag

object Tools {
  def Error[T: ClassTag](tensor1: Tensor[T], tensor2: Tensor[T])(
    implicit ev: TensorNumeric[T]): Double = {
    require(tensor1.nElement() == tensor2.nElement())
    var ret = 0.0
    for (i <- 0 until tensor1.nElement()) {
      ret += math.abs(
        ev.toType[Double](tensor1.storage().array()(i)) -
          ev.toType[Double](tensor2.storage().array()(i)))
    }
    ret
  }

  def CumulativeError[T: ClassTag](tensor1: Tensor[T], tensor2: Tensor[T], msg: String)(
      implicit ev: TensorNumeric[T]): Double = {
    val ret = Error[T](tensor1, tensor2)
    println(msg.toUpperCase + " CUMULATIVE ERROR: " + ret)
    ret
  }

  def AverageError[T: ClassTag](tensor1: Tensor[T], tensor2: Tensor[T], msg: String)(
    implicit ev: TensorNumeric[T]): Double = {
    require(tensor1.nElement() > 0)
    val ret = Error[T](tensor1, tensor2) / tensor1.nElement()
    println(msg.toUpperCase + " AVERAGE ERROR: " + ret)
    ret
  }

  def GetRandTimes(): Int = 10
}
