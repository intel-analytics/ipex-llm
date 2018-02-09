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

import java.util.{ArrayList => JArrayList, HashMap => JHashMap, List => JList, Map => JMap}

import com.intel.analytics.bigdl.nn.SpatialBatchNormalization
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.keras._
import com.intel.analytics.bigdl.numeric._
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.collection.JavaConverters._
import scala.language.existentials
import scala.reflect.ClassTag


object PythonBigDLKeras {

  def ofFloat(): PythonBigDLKeras[Float] = new PythonBigDLKeras[Float]()

  def ofDouble(): PythonBigDLKeras[Double] = new PythonBigDLKeras[Double]()
}

class PythonBigDLKeras[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonBigDL[T] {

  def toScalaShape(inputShape: JList[Int]): Shape = {
    if (inputShape == null) {
      null
    } else {
      Shape(inputShape.asScala.toArray)
    }
  }

  def toScalaMultiShape(inputShape: JList[JList[Int]]): Shape = {
    if (inputShape == null) {
      null
    } else {
      Shape(inputShape.asScala.toArray.map(shape => Shape(shape.asScala.toArray)).toList)
    }
  }

  def createKerasInputLayer(
    inputShape: JList[Int] = null): Input[T] = {
    InputLayer(inputShape = toScalaShape(inputShape))
  }

  def createKerasDense(
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

  def createKerasEmbedding(
    inputDim: Int,
    outputDim: Int,
    init: String = "uniform",
    wRegularizer: Regularizer[T] = null,
    inputShape: JList[Int] = null): Embedding[T] = {
    Embedding[T](inputDim, outputDim, init, wRegularizer, toScalaShape(inputShape))
  }

  def createKerasBatchNormalization(
    epsilon: Double = 0.001,
    momentum: Double = 0.99,
    betaInit: String = "zero",
    gammaInit: String = "one",
    dimOrdering: String = "th",
    inputShape: JList[Int] = null): BatchNormalization[T] = {
    BatchNormalization[T](epsilon, momentum, betaInit,
      gammaInit, dimOrdering, toScalaShape(inputShape))
  }

  def setKerasRunningMean(module: BatchNormalization[T], runningMean: JTensor): Unit = {
    module.labor.asInstanceOf[SpatialBatchNormalization[T]]
      .runningMean.set(toTensor(runningMean))
  }

  def setKerasRunningStd(module: BatchNormalization[T], runningStd: JTensor): Unit = {
    module.labor.asInstanceOf[SpatialBatchNormalization[T]]
      .runningVar.set(toTensor(runningStd))
  }

  def getKerasRunningMean(module: BatchNormalization[T]): JTensor = {
    toJTensor(module.labor.asInstanceOf[SpatialBatchNormalization[T]]
      .runningMean)
  }

  def getKerasRunningStd(module: BatchNormalization[T]): JTensor = {
    toJTensor(module.labor.asInstanceOf[SpatialBatchNormalization[T]]
      .runningVar)
  }

  def createKerasMerge(
    layers: JList[AbstractModule[Activity, Activity, T]] = null,
    mode: String = "sum",
    concatAxis: Int = -1,
    inputShape: JList[JList[Int]]): Merge[T] = {
    Merge[T](layers.asScala.toList, mode, concatAxis, toScalaMultiShape(inputShape))
  }

}
