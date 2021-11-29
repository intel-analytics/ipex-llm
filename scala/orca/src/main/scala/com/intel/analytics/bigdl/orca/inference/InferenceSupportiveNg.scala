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

package com.intel.analytics.bigdl.orca.inference

import com.intel.analytics.bigdl.dllib.feature.dataset.Sample
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import java.util.{List => JList}
import java.lang.{Float => JFloat}
import java.lang.{Integer => JInt}
import java.util

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dllib.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.{T, Table}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag


trait InferenceSupportiveNg {

  def timing[T](name: String)(f: => T): T = {
    val begin = System.currentTimeMillis
    val result = f
    val end = System.currentTimeMillis
    val cost = (end - begin)
    InferenceSupportive.logger.info(s"$name time elapsed [${cost / 1000} s, ${cost % 1000} ms].")
    result
  }

  def tensorToJTensor(tensor: Tensor[Float]): JTensor = {
    new JTensor(tensor.storage().array(), tensor.size())
  }

  def tableToJTensorList(table: Table): JList[JTensor] = {
    val jTensorList = new util.ArrayList[JTensor]()
    table.toSeq[Tensor[Float]].foreach(t =>
      jTensorList.add(tensorToJTensor(t))
    )
    jTensorList
  }

  def jTensorListToActivity(inputs: JList[JTensor]): Activity = {
    val tensorArray = inputs.asScala.map(jt => {
      Tensor[Float](jt.getData, jt.getShape)
    })
    if (tensorArray.size == 1) {
      tensorArray(0)
    } else {
      T.array(tensorArray.toArray)
    }

  }

  def transferTensorsToTensorOfBatch(tensors: Array[JTensor]): Tensor[Float] = {
    val batchSize = tensors.length
    val head = tensors.head
    val shape = (ArrayBuffer(batchSize) ++= head.getShape)
    val data = ArrayBuffer[Float]()
    var i = 0
    while (i < batchSize) {
      val tensor = tensors(i)
      val tensorData = tensor.getData
      data ++= tensorData
      i += 1
    }
    require(data.length == shape.reduce(_ * _),
      "data length should be equal to the product of shape")
    Tensor[Float](data.toArray, shape.toArray)
  }

  def makeMetaModel(original: AbstractModule[Activity, Activity, Float]):
  AbstractModule[Activity, Activity, Float] = {
    val metaModel = original.cloneModule()
    releaseWeightBias(metaModel)
    metaModel
  }

  def clearWeightBias(model: Module[Float]): Unit = {
    model.reset()
    val weightBias = model.parameters()._1
    val clonedWeightBias = model.parameters()._1.map(tensor => {
      val newTensor = Tensor[Float]().resizeAs(tensor)
      newTensor.copy(tensor)
    })
    val localWeightBias = model.parameters()._1
    var i = 0
    while (i < localWeightBias.length) {
      if (localWeightBias(i) != null) {
        localWeightBias(i).set(clonedWeightBias(i))
      }
      i += 1
    }
    releaseTensors(model.parameters()._1)
    releaseTensors(model.parameters()._2)
  }

  def releaseWeightBias(model: Module[Float]): Unit = {
    model.reset()
    releaseTensors(model.parameters()._1)
    releaseTensors(model.parameters()._2)
  }

  private def releaseTensors[T: ClassTag](tensors: Array[Tensor[T]])
                                         (implicit ev: TensorNumeric[T]) = {
    var i = 0
    while (i < tensors.length) {
      if (tensors(i) != null) {
        tensors(i).set()
      }
      i += 1
    }
  }

  def makeUpModel(clonedModel: Module[Float], weightBias: Array[Tensor[Float]]):
  AbstractModule[Activity, Activity, Float] = {
    putWeightBias(clonedModel, weightBias)
    clonedModel.evaluate()
    clonedModel
  }

  private def putWeightBias(target: Module[Float], weightBias: Array[Tensor[Float]]):
  Module[Float] = {
    val localWeightBias = target.parameters()._1
    var i = 0
    while (i < localWeightBias.length) {
      if (localWeightBias(i) != null) {
        localWeightBias(i).set(weightBias(i))
      }
      i += 1
    }
    target
  }
}

object InferenceSupportiveNg {
  val logger = LoggerFactory.getLogger(getClass)
  val modelType = List(
    "frozenModel",
    "savedModel"
  )
}

