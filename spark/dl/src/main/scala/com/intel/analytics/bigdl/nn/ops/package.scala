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
package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

package object ops {
  object Add {
    def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): Operation[Table, T]
    = ModuleToOperation[Table, T](CAddTable())
  }

  object Subtract {
    def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): Operation[Table, T]
    = ModuleToOperation[Table, T](CSubTable())
  }

  object Multiply {
    def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): Operation[Table, T]
    = ModuleToOperation[Table, T](CMulTable())
  }

  object Divide {
    def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): Operation[Table, T]
    = ModuleToOperation[Table, T](CDivTable())
  }

  object Sum {
    def apply[T: ClassTag](axis: Int, keepDim: Boolean = false)
      (implicit ev: TensorNumeric[T]): Operation[Tensor[T], T]
    = ModuleToOperation[Tensor[T], T](
      com.intel.analytics.bigdl.nn.Sum(dimension = axis, squeeze = !keepDim))
  }

  object Reshape {
    def apply[T: ClassTag](size: Array[Int])
      (implicit ev: TensorNumeric[T]): Operation[Tensor[T], T]
    = ModuleToOperation[Tensor[T], T](
      com.intel.analytics.bigdl.nn.InferReshape(size: Array[Int]))
  }

  object Squeeze {
    def apply[T: ClassTag](axis: Array[Int] = null)
      (implicit ev: TensorNumeric[T]): Operation[Tensor[T], T]
    = ModuleToOperation[Tensor[T], T](
      com.intel.analytics.bigdl.nn.Squeeze(dims = axis, batchMode = false))
  }

  object Identity {
    def apply[T: ClassTag]()
      (implicit ev: TensorNumeric[T]): Operation[Activity, T]
    = ModuleToOperation[Activity, T](
      com.intel.analytics.bigdl.nn.Identity()
        .asInstanceOf[AbstractModule[Activity, Tensor[T], T]])
  }

  object ReLU {
    def apply[T: ClassTag]()
      (implicit ev: TensorNumeric[T]): Operation[Tensor[T], T]
    = ModuleToOperation[Tensor[T], T](
      com.intel.analytics.bigdl.nn.ReLU())
  }
}
