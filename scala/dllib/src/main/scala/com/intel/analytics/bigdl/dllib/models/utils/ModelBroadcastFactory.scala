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
package com.intel.analytics.bigdl.models.utils

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Create a ModelBroadcast. User can override how model is broadcast by extending this class to
 * create a customized ModelBroadcast. To enable customized broadcast factory, user need to specify
 * its full class name via system property bigdl.ModelBroadcastFactory.
 */
trait ModelBroadcastFactory {
  def create[T: ClassTag]()(implicit ev: TensorNumeric[T]) : ModelBroadcast[T]
}

private[bigdl] class DefaultModelBroadcastFactory extends ModelBroadcastFactory {
  override def create[T: ClassTag]()(implicit ev: TensorNumeric[T]): ModelBroadcast[T] = {
    new ModelBroadcastImp[T]()
  }
}

private[bigdl] class ProtoBufferModelBroadcastFactory extends ModelBroadcastFactory {
  override def create[T: ClassTag]()(implicit ev: TensorNumeric[T]): ModelBroadcast[T] = {
    new ModelBroadcastImp[T](true)
  }
}
