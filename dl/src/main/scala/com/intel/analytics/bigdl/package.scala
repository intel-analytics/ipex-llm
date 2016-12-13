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
package com.intel.analytics

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.language.implicitConversions

package object bigdl {
  type Module[T] = com.intel.analytics.bigdl.nn.abstractnn.Module[Activity, Activity, T]
  type Criterion[T] = com.intel.analytics.bigdl.nn.abstractnn.Criterion[Activity, Activity, T]

  implicit def convModule[T](
    module: com.intel.analytics.bigdl.nn.abstractnn.Module[_, _, T]
  ): Module[T] = module.asInstanceOf[Module[T]]

  implicit def convCriterion[T](
    criterion: com.intel.analytics.bigdl.nn.abstractnn.Criterion[_, _, T]
  ): Criterion[T] = criterion.asInstanceOf[Criterion[T]]

  val numeric = com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
}
