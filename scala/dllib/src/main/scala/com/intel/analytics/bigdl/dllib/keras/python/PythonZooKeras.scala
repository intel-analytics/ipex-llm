/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.pipeline.api.keras.python

import java.util.{List => JList}
import scala.collection.JavaConverters._

import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.python.api.PythonBigDLKeras
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.zoo.pipeline.api.keras.layers._

import scala.reflect.ClassTag

object PythonZooKeras {

  def ofFloat(): PythonZooKeras[Float] = new PythonZooKeras[Float]()

  def ofDouble(): PythonZooKeras[Double] = new PythonZooKeras[Double]()
}

class PythonZooKeras[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonBigDLKeras[T] {

  def createZooKerasModel(
      input: JList[ModuleNode[T]],
      output: JList[ModuleNode[T]]): Model[T] = {
    Model[T](input.asScala.toArray, output.asScala.toArray)
  }

  def createZooKerasSequential(): Sequential[T] = {
    Sequential[T]()
  }

  def createZooKerasInput(
      name : String = null,
      inputShape: JList[Int] = null): ModuleNode[T] = {
    Input(name = name, inputShape = toScalaShape(inputShape))
  }

  def createZooKerasInputLayer(
      inputShape: JList[Int] = null): KerasLayer[Activity, Activity, T] = {
    InputLayer(inputShape = toScalaShape(inputShape))
  }

  def createZooKerasDense(
      outputDim: Int,
      init: String = "glorot_uniform",
      activation: String = null,
      wRegularizer: Regularizer[T] = null,
      bRegularizer: Regularizer[T] = null,
      bias: Boolean = true,
      inputShape: JList[Int] = null): Dense[T] = {
    Dense(outputDim, init, activation, wRegularizer,
      bRegularizer, bias, toScalaShape(inputShape))
  }

}
