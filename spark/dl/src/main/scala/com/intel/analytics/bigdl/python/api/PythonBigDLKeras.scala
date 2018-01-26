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

package com.intel.analytics.bigdl.python.api

import java.lang.{Boolean => JBoolean}
import java.util.{ArrayList => JArrayList, HashMap => JHashMap, List => JList, Map => JMap}

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.nn.keras.Dense
import com.intel.analytics.bigdl.nn.{InitializationMethod, Linear, RandomUniform}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Shape, Table}

import scala.collection.JavaConverters._
import scala.collection.mutable.Map
import scala.language.existentials
import scala.reflect.ClassTag


trait PythonBigDLKeras {
  self: PythonBigDL[T: ClassTag] =>
  def createDense(outputDim: Int,
                   init: InitializationMethod = RandomUniform,
                   activation: TensorModule[T] = null,
                   wRegularizer: Regularizer[T] = null,
                   bRegularizer: Regularizer[T] = null,
                   bias: Boolean = true,
                   inputShape: Shape = null): Dense[T] = {
    Dense(outputDim,
      init,
      activation,
      wRegularizer,
      bRegularizer,
      bias,
      inputShape)
  }
}
