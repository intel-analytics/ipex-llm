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

package com.intel.analytics.bigdl.nn.abstractnn

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect._

trait Activity {
  def toTensor[T]: Tensor[T] = this match {
    case tensor: Tensor[T] => tensor
    case table: Table => throw
      new IllegalArgumentException("Table cannot be cast to Tensor")
    case _ => throw
      new IllegalArgumentException("Activity only support tensor and table now")
  }

  def toTable: Table = this match {
    case table: Table => table
    case tensor: Tensor[_] => throw
      new IllegalArgumentException("Tensor cannot be cast to Table")
    case _ => throw
      new IllegalArgumentException("Activity only support tensor and table now")
  }
}

object Activity {
  def apply[A <: Activity: ClassTag, T : ClassTag]()(
    implicit ev: TensorNumeric[T]): A = {
    val result = if (classTag[A] == classTag[Tensor[T]]) {
      Tensor[T]()
    } else if (classTag[A] == classTag[Table]) {
      T()
    } else {
      null
    }

    result.asInstanceOf[A]
  }
}
