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
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils
import com.intel.analytics.zoo.pipeline.api.keras.models.{Model, Sequential}

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


  // ================================= Torch layers in Keras style =================================

  def createZooKerasSelect(
      dim: Int,
      index: Int,
      inputShape: JList[Int] = null): Select[T] = {
    Select(dim, index, toScalaShape(inputShape))
  }

  def createZooKerasNarrow(
      dim: Int,
      offset: Int,
      length: Int = 1,
      inputShape: JList[Int] = null): Narrow[T] = {
    Narrow(dim, offset, length, toScalaShape(inputShape))
  }

  def createZooKerasSqueeze(
      dims: JList[Int],
      inputShape: JList[Int] = null): Squeeze[T] = {
    Squeeze(toScalaArray(dims), toScalaShape(inputShape))
  }

  def createZooKerasAddConstant(
      constant: Double,
      inputShape: JList[Int] = null): AddConstant[T] = {
    AddConstant(constant, toScalaShape(inputShape))
  }

  def createZooKerasMulConstant(
      constant: Double,
      inputShape: JList[Int] = null): MulConstant[T] = {
    MulConstant(constant, toScalaShape(inputShape))
  }

  def createZooKerasLRN2D(
      alpha: Double = 1e-4,
      k: Double = 1.0,
      beta: Double = 0.75,
      n: Int = 5,
      dimOrdering: String = "th",
      inputShape: JList[Int] = null): LRN2D[T] = {
    LRN2D(alpha, k, beta, n, dimOrdering, toScalaShape(inputShape))
  }

  def createZooKerasShareConvolution2D(
      nbFilter: Int,
      nbRow: Int,
      nbCol: Int,
      init: String = "glorot_uniform",
      activation: String = null,
      subsample: JList[Int],
      padH: Int = 0,
      padW: Int = 0,
      propagateBack: Boolean = true,
      dimOrdering: String = "th",
      wRegularizer: Regularizer[T] = null,
      bRegularizer: Regularizer[T] = null,
      bias: Boolean = true,
      inputShape: JList[Int] = null): ShareConvolution2D[T] = {
    new ShareConvolution2D(nbFilter, nbRow, nbCol, KerasUtils.getInitMethod(init),
      KerasUtils.getKerasActivation(activation), toScalaArray(subsample),
      padH, padW, propagateBack, KerasUtils.toBigDLFormat(dimOrdering),
      wRegularizer, bRegularizer, bias, toScalaShape(inputShape))
  }

}
