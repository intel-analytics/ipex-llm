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
import com.intel.analytics.bigdl.nn.tf.TensorModuleWrapper
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

package object tf {

  object Mean {
    def apply[T: ClassTag, D: ClassTag](dimension: Int = 1,
                                        nInputDims: Int = -1,
                                        squeeze: Boolean = true)
                                       (implicit ev: TensorNumeric[T], evd: TensorNumeric[D]):
    AbstractModule[Activity, Activity, T]
    = TensorModuleWrapper[T, D](
      com.intel.analytics.bigdl.nn.Mean(dimension, nInputDims, squeeze))
  }

  object Abs {
    def apply[T: ClassTag, D: ClassTag]()
                                       (implicit ev: TensorNumeric[T], evd: TensorNumeric[D]):
    AbstractModule[Activity, Activity, T]
    = TensorModuleWrapper[T, D](
      com.intel.analytics.bigdl.nn.Abs[D]())
  }

  object Clamp {
    def apply[T: ClassTag, D: ClassTag](min: Int, max: Int)
                                       (implicit ev: TensorNumeric[T], evd: TensorNumeric[D]):
    AbstractModule[Activity, Activity, T]
    = TensorModuleWrapper[T, D](
      com.intel.analytics.bigdl.nn.Clamp[D](min, max))
  }

  object ReLU6 {
    def apply[T: ClassTag, D: ClassTag]()
                                       (implicit ev: TensorNumeric[T], evd: TensorNumeric[D]):
    AbstractModule[Activity, Activity, T]
    = TensorModuleWrapper[T, D](
      com.intel.analytics.bigdl.nn.ReLU6[D]())
  }

  object ELU {
    def apply[T: ClassTag, D: ClassTag]()
                                       (implicit ev: TensorNumeric[T], evd: TensorNumeric[D]):
    AbstractModule[Activity, Activity, T]
    = TensorModuleWrapper[T, D](
      com.intel.analytics.bigdl.nn.ELU[D]())
  }

  object Log {
    def apply[T: ClassTag, D: ClassTag]()
                                       (implicit ev: TensorNumeric[T], evd: TensorNumeric[D]):
    AbstractModule[Activity, Activity, T]
    = TensorModuleWrapper[T, D](
      com.intel.analytics.bigdl.nn.Log[D]())
  }

  object Power {
    def apply[T: ClassTag, D: ClassTag](power: Double,
                                        scale : Double = 1,
                                        shift : Double = 0)
                                       (implicit ev: TensorNumeric[T], evd: TensorNumeric[D]):
    AbstractModule[Activity, Activity, T]
    = TensorModuleWrapper[T, D](
      com.intel.analytics.bigdl.nn.Power[D](power, scale, shift))
  }

  object SoftPlus {
    def apply[T: ClassTag, D: ClassTag]()
                                       (implicit ev: TensorNumeric[T], evd: TensorNumeric[D]):
    AbstractModule[Activity, Activity, T]
    = TensorModuleWrapper[T, D](
      com.intel.analytics.bigdl.nn.SoftPlus[D]())
  }

  object SoftSign {
    def apply[T: ClassTag, D: ClassTag]()
                                       (implicit ev: TensorNumeric[T], evd: TensorNumeric[D]):
    AbstractModule[Activity, Activity, T]
    = TensorModuleWrapper[T, D](
      com.intel.analytics.bigdl.nn.SoftSign[D]())
  }
}
