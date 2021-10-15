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

package com.intel.analytics.bigdl.dllib.net.python

import java.util.{ArrayList, List => JList}

import com.intel.analytics.bigdl.dllib.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.dllib.nn.keras.KerasLayer
import com.intel.analytics.bigdl.dllib.utils.python.api.JTensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.common.PythonZoo
import com.intel.analytics.bigdl.dllib.utils.python.api.EvaluatedResult
import com.intel.analytics.bigdl.dllib.keras.Net
import com.intel.analytics.bigdl.dllib.net.NetUtils

import scala.collection.JavaConverters._
import scala.reflect.ClassTag


object PythonDllibNet {

  def ofFloat(): PythonDllibNet[Float] = new PythonDllibNet[Float]()

  def ofDouble(): PythonDllibNet[Double] = new PythonDllibNet[Double]()

}

class PythonDllibNet[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonZoo[T] {

  def newGraph(model: NetUtils[T, _],
               outputs: JList[String]): NetUtils[T, _] = {
    model.newGraph(outputs.asScala).asInstanceOf[NetUtils[T, _]]
  }

  def freezeUpTo(model: NetUtils[T, _], names: JList[String]): Unit = {
    model.freezeUpTo(names.asScala: _*)
  }

  def netLoadBigDL(
                    modulePath: String,
                    weightPath : String): AbstractModule[Activity, Activity, T] = {
    Net.loadBigDL[T](modulePath, weightPath)
  }

  def netLoadCaffe(
                    defPath: String,
                    modelPath : String): AbstractModule[Activity, Activity, T] = {
    Net.loadCaffe[T](defPath, modelPath)
  }

  def netLoad(
               modulePath: String,
               weightPath : String): AbstractModule[Activity, Activity, T] = {
    Net.load[T](modulePath, weightPath)
  }

  def netLoadTorch(
                    path: String): AbstractModule[Activity, Activity, T] = {
    Net.loadTorch[T](path)
  }

  def netToKeras(value: NetUtils[T, _]): KerasLayer[Activity, Activity, T] = {
    value.toKeras()
  }
}
