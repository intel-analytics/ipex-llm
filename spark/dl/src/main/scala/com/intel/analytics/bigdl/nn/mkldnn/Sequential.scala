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
package com.intel.analytics.bigdl.nn.mkldnn

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.{Sequential => Seq}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

class Sequential[T: ClassTag](implicit ev: TensorNumeric[T]) extends Seq[T] with MklDnnContainer {
  override def add(module: AbstractModule[_ <: Activity, _ <: Activity, T]): this.type = {
    require(executions == null, "You should not call add after compilation")
    require(module.isInstanceOf[MklDnnModule], "layer should be MklDnnModule")
    super.add(module)
  }

  override private[mkldnn] def initPrimitives(runtime: MklDnnRuntime) = {
    executions.foreach(m => {
      m.phase = phase
      m.initPrimitives(runtime)
    })
    modules.clear()
    modules.appendAll(executions.map(_.asInstanceOf[AbstractModule[Activity, Activity, T]]))
  }

  override private[mkldnn] def inferOutputFormats(): Array[Memory] = {
    var lastOutputFormats = expectInputFormats
    val executionsBuffer = new ArrayBuffer[MklDnnModule]()
    modules.foreach(m => {
      require(m.isInstanceOf[MklDnnModule], "layer should be MklDnnModule")
      val _m = m.asInstanceOf[MklDnnModule]

      require(Memory.isCompatible(lastOutputFormats, _m.expectInputFormats),
        "memory format is not compatible with expected one")
      _m.setInputFormats(lastOutputFormats)
      lastOutputFormats = _m.inferOutputFormats()
      executionsBuffer.append(_m)
    })
    executions = executionsBuffer.toArray
    lastOutputFormats
  }

  override private[mkldnn] def setInputFormats(formats: Array[Memory]): Unit = {
    modules(0).asInstanceOf[MklDnnModule].setInputFormats(formats)
  }

  override private[mkldnn] def expectInputFormats = {
    modules(0).asInstanceOf[MklDnnModule].expectInputFormats
  }
}
