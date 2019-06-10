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

package com.intel.analytics.bigdl.utils.intermediate

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.ReflectionUtils

import scala.reflect.ClassTag

private[bigdl] class BlasToIR[T: ClassTag] extends ConvertBase[Module[T], IRElement[T]]{

  private def className(layer: Module[T]): String = {
    val name = layer.getClass.getSimpleName
    s"com.intel.analytics.bigdl.utils.intermediate.IR$name"
  }

  // reminder: some undefined IR operations can be presented by IRGeneralModule
  override def convertLayerCheck(layer: Module[T]): Boolean = {
    ReflectionUtils.findClass(className(layer)) != null ||
    layer.isInstanceOf[AbstractModule[Activity, Activity, T]]
  }

  override def convertLayer(layer : Module[T]) : IRElement[T] = {
    val cls = ReflectionUtils.findClass(className(layer))
    if ( cls != null) {
      ReflectionUtils.reflectToIR(layer, cls)
    } else if (layer.isInstanceOf[AbstractModule[Activity, Activity, T]]) {
      val op = IRGeneralModule[T](
        layer.asInstanceOf[AbstractModule[Activity, Activity, T]])
      IRElement(layer.getName(), op)
    } else {
      throw new UnsupportedOperationException(s"can not convert $layer to IRelement ")
    }
  }
}

private[bigdl] object BlasToIR {
  def apply[T: ClassTag](implicit ev: TensorNumeric[T]): BlasToIR[T] = new BlasToIR
}
