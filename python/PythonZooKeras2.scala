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

import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.python.api.PythonBigDLKeras
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.pipeline.api.keras2.layers.{Dense, Conv1D}

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

}

