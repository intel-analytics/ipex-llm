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

package com.intel.analytics.bigdl.nn.dnn

import com.intel.analytics.bigdl.tensor.Tensor

import scala.reflect.ClassTag

object Utils {
  def computeOutput(input: Long, pad: Long, kernel: Long, stride: Long): Long = {
    (input + 2 * pad - kernel) / stride + 1
  }

  def getSize[T: ClassTag](tensor: Tensor[T], dimension: Int): Array[Long] = {
    (tensor.size() ++ Array.fill(dimension - tensor.dim())(1)).reverse.map(_.toLong)
  }

  def reverseAndToInt(array: Array[Long]): Array[Int] = {
    array.reverse.map(_.toInt)
  }
}
