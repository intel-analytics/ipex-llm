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
package com.intel.analytics.bigdl.nn

import java.nio.ByteOrder

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{CaffeLoader, File}
import com.intel.analytics.bigdl.utils.tf.{TensorflowDataFormat, TensorflowLoader}

import scala.reflect.ClassTag

object Module {
  def load[T: ClassTag](path : String) : AbstractModule[Activity, Activity, T] = {
    File.load[AbstractModule[Activity, Activity, T]](path)
  }

  def loadTorch[T: ClassTag](path : String) : AbstractModule[Activity, Activity, T] = {
    File.loadTorch[AbstractModule[Activity, Activity, T]](path)
  }

  def loadCaffe[T: ClassTag](model: AbstractModule[Activity, Activity, T],
    defPath: String, modelPath: String, matchAll: Boolean = true)(
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Activity, T] = {
    CaffeLoader.load[T](model, defPath, modelPath, matchAll)
  }

  /**
   * Load tensorflow model from its saved protobuf file.
   * @param file where is the protobuf model file
   * @param inputs input node names
   * @param outputs output node names, the output tensor order is same with the node order
   * @param byteOrder byte order in the tensorflow file. The default value is little endian
   * @return BigDL model
   */
  def loadTF[T: ClassTag](file: String, inputs: Seq[String], outputs: Seq[String],
            byteOrder: ByteOrder = ByteOrder.LITTLE_ENDIAN)(
    implicit ev: TensorNumeric[T]): Module[T] = {

    TensorflowLoader.load(file, inputs, outputs, byteOrder)
  }

  def flatten[@specialized(Float, Double) T: ClassTag](parameters: Array[Tensor[T]])(
    implicit ev: TensorNumeric[T]): Tensor[T] = {
    val compactedTensor = isCompact(parameters)
    if (compactedTensor != null) {
      return compactedTensor
    }
    var i = 0
    var length = 0
    while (i < parameters.length) {
      require(parameters(i).isContiguous())
      length += parameters(i).nElement()
      i += 1
    }

    val result = Tensor[T](length)
    val resultStorage = result.storage()

    i = 0
    var offset = 0
    while (i < parameters.length) {
      System.arraycopy(parameters(i).storage().array(), parameters(i).storageOffset() - 1,
        resultStorage.array(), offset, parameters(i).nElement())
      parameters(i).set(resultStorage, offset + 1, parameters(i).size(), parameters(i).stride())
      offset += parameters(i).nElement()
      i += 1
    }

    result
  }

  def isCompact[@specialized(Float, Double) T: ClassTag](paramters: Array[Tensor[T]])(
    implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(paramters.length > 0)
    var i = 1
    val storage = paramters(0).storage()
    var length = paramters(0).nElement()
    while (i < paramters.length) {
      if (!storage.eq(paramters(i).storage())) {
        return null
      }
      length += paramters(i).nElement()
      i += 1
    }

    if (length != storage.array().length) {
      return null
    }

    return Tensor(storage)
  }
}
