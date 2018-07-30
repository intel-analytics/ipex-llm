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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.python.api.PythonBigDLKeras
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.autograd.Variable
import com.intel.analytics.zoo.pipeline.api.keras2.layers.{Conv1D, Dense, MaxPooling1D, AveragePooling1D, Maximum, Minimum, Average}
import scala.collection.JavaConverters._
import com.intel.analytics.zoo.pipeline.api.keras2.layers._

import scala.reflect.ClassTag

object PythonZooKeras2 {

  def ofFloat(): PythonZooKeras2[Float] = new PythonZooKeras2[Float]()

  def ofDouble(): PythonZooKeras2[Double] = new PythonZooKeras2[Double]()
}

class PythonZooKeras2[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonBigDLKeras[T] {

  def createZooKeras2Dense(
      units: Int,
      kernelInitializer: String = "glorot_uniform",
      biasInitializer: String = "zero",
      activation: String = null,
      kernelRegularizer: Regularizer[T] = null,
      biasRegularizer: Regularizer[T] = null,
      useBias: Boolean = true,
      inputShape: JList[Int] = null): Dense[T] = {
     Dense[T](
      units = units,
      kernelInitializer = kernelInitializer,
      biasInitializer = biasInitializer,
      activation = activation,
      kernelRegularizer = kernelRegularizer,
      biasRegularizer = biasRegularizer,
      useBias = useBias,
      inputShape = toScalaShape(inputShape))
  }

  def createZooKeras2Conv1D(
      filters: Int,
      kernelSize: Int,
      strides: Int = 1,
      padding: String = "valid",
      activation: String = null,
      useBias: Boolean = true,
      kernelInitializer: String = "glorot_uniform",
      biasInitializer: String = "zero",
      kernelRegularizer: Regularizer[T] = null,
      biasRegularizer: Regularizer[T] = null,
      inputShape: JList[Int] = null): Conv1D[T] = {
    Conv1D(
      filters,
      kernelSize,
      strides,
      padding,
      activation,
      useBias,
      kernelInitializer,
      biasInitializer,
      kernelRegularizer,
      biasRegularizer,
      toScalaShape(inputShape))
  }

  def createZooKeras2Conv2D(
      filters: Int,
      kernelSize: JList[Int],
      strides: JList[Int],
      padding: String = "valid",
      dataFormat: String = "channels_first",
      activation: String = null,
      useBias: Boolean = true,
      kernelInitializer: String = "glorot_uniform",
      biasInitializer: String = "zero",
      kernelRegularizer: Regularizer[T] = null,
      biasRegularizer: Regularizer[T] = null,
      inputShape: JList[Int] = null): Conv2D[T] = {
    Conv2D(
      filters,
      kernelSize.asScala.toArray,
      strides.asScala.toArray,
      padding,
      dataFormat,
      activation,
      useBias,
      kernelInitializer,
      biasInitializer,
      kernelRegularizer,
      biasRegularizer,
      toScalaShape(inputShape))
  }

  def createZooKeras2MaxPooling1D(
      poolSize: Int = 2,
      strides: Int = -1,
      padding: String = "valid",
      inputShape: JList[Int] = null): MaxPooling1D[T] = {
    MaxPooling1D(
      poolSize,
      strides,
      padding,
      toScalaShape(inputShape))
  }

  def createZooKeras2AveragePooling1D(
      poolSize: Int = 2,
      strides: Int = -1,
      padding: String = "valid",
      inputShape: JList[Int] = null): AveragePooling1D[T] = {
    AveragePooling1D(
      poolSize,
      strides,
      padding,
      toScalaShape(inputShape))
  }

  def createZooKeras2Maximum(
      inputShape: JList[Int] = null): Maximum[T] = {
    Maximum(
      toScalaShape(inputShape))
  }

  def createZooKeras2Minimum(
      inputShape: JList[Int] = null): Minimum[T] = {
    Minimum(
      toScalaShape(inputShape))
  }

  def createZooKeras2Average(
      inputShape: JList[Int] = null): Average[T] = {
    Average(
      toScalaShape(inputShape))
  }

  def createZooKeras2GlobalAveragePooling1D(
      inputShape: JList[Int] = null): GlobalAveragePooling1D[T] = {
    GlobalAveragePooling1D(
      toScalaShape(inputShape))
  }

  def createZooKeras2GlobalMaxPooling1D(
      inputShape: JList[Int] = null): GlobalMaxPooling1D[T] = {
    GlobalMaxPooling1D(
      toScalaShape(inputShape))
  }

  def createZooKeras2GlobalAveragePooling2D(
      dataFormat: String = "channels_first",
      inputShape: JList[Int] = null): GlobalAveragePooling2D[T] = {
    GlobalAveragePooling2D(
      dataFormat,
      toScalaShape(inputShape))
  }

  def createZooKeras2Activation(
      activation: String,
      inputShape: JList[Int] = null): Activation[T] = {
    Activation(
      activation,
      toScalaShape(inputShape))
  }

  def createZooKeras2Dropout(
      rate: Double,
      inputShape: JList[Int] = null): Dropout[T] = {
    Dropout(rate, toScalaShape(inputShape))
  }

  def createZooKeras2Flatten(
      inputShape: JList[Int] = null): Flatten[T] = {
    Flatten(toScalaShape(inputShape))
  }

  def createZooKeras2Cropping1D(
      cropping: JList[Int],
      inputShape: JList[Int] = null): Cropping1D[T] = {
    new Cropping1D(toScalaArray(cropping), toScalaShape(inputShape))
  }

  def createZooKeras2LocallyConnected1D(
      filters: Int,
      kernelSize: Int,
      strides: Int = 1,
      padding: String = "valid",
      activation: String = null,
      kernelRegularizer: Regularizer[T] = null,
      biasRegularizer: Regularizer[T] = null,
      useBias: Boolean = true,
      inputShape: JList[Int] = null): LocallyConnected1D[T] = {
    LocallyConnected1D(filters, kernelSize, strides, padding, activation,
      kernelRegularizer, biasRegularizer, useBias, toScalaShape(inputShape))
  }



}
