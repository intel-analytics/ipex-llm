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
import com.intel.analytics.bigdl.utils.File
import com.intel.analytics.bigdl.utils.caffe.CaffeLoader
import com.intel.analytics.bigdl.utils.serializer.ModuleLoader
import com.intel.analytics.bigdl.utils.tf.{Session, TensorflowDataFormat, TensorflowLoader}

import scala.reflect.ClassTag

object Module {
  /**
   * Load model from path.
   *
   * @param path path to save module, local file system, HDFS and Amazon S3 is supported.
   *             HDFS path should be like "hdfs://[host]:[port]/xxx"
   *             Amazon S3 path should be like "s3a://bucket/xxx"
   * @tparam T numeric type
   * @return model loaded from path
   */
  @deprecated("Java based serialization not recommended any more, please use loadModule instead",
    "0.3")
  def load[T: ClassTag](path : String) : AbstractModule[Activity, Activity, T] = {
    File.load[AbstractModule[Activity, Activity, T]](path)
  }

  /**
   * Load model from path.
   *
   * @param path path to save module, local file system, HDFS and Amazon S3 is supported.
   *             HDFS path should be like "hdfs://[host]:[port]/xxx"
   *             Amazon S3 path should be like "s3a://bucket/xxx"
   * @param weightPath : where weight is stored
   * @tparam T numeric type
   * @return model loaded from path
   */
  def loadModule[T: ClassTag](path : String,
    weightPath : String = null)(implicit ev: TensorNumeric[T])
  : AbstractModule[Activity, Activity, T] = {
    ModuleLoader.loadFromFile(path, weightPath)
  }

  def loadTorch[T: ClassTag](path : String) : AbstractModule[Activity, Activity, T] = {
    File.loadTorch[AbstractModule[Activity, Activity, T]](path)
  }

  @deprecated("Please try to use the loadCaffeModel API", "0.2")
  def loadCaffe[T: ClassTag](model: AbstractModule[Activity, Activity, T],
    defPath: String, modelPath: String, matchAll: Boolean = true)(
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Activity, T] = {
    CaffeLoader.load[T](model, defPath, modelPath, matchAll)
  }

  /**
   * Loaf caffe trained model from prototxt and weight files
   * @param defPath  caffe model definition file path
   * @param modelPath caffe model binary file containing weight and bias
   */
  def loadCaffeModel[T: ClassTag](defPath: String, modelPath: String)(
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Activity, T] = {
    CaffeLoader.loadCaffe[T](defPath, modelPath)._1
  }
  /**
   * Load tensorflow model from its saved protobuf file.
   * @param graphFile where is the protobuf model file
   * @param inputs input node names
   * @param outputs output node names, the output tensor order is same with the node order
   * @param byteOrder byte order in the tensorflow file. The default value is little endian
   * @param binFile where is the model variable file
   * @param generatedBackward if generate backward graph
   * @return BigDL model
   */
  def loadTF[T: ClassTag](graphFile: String, inputs: Seq[String], outputs: Seq[String],
        byteOrder: ByteOrder = ByteOrder.LITTLE_ENDIAN,
        binFile: Option[String] = None, generatedBackward: Boolean = true)(
        implicit ev: TensorNumeric[T]): Module[T] = {
    TensorflowLoader.load(graphFile, inputs, outputs, byteOrder, binFile, generatedBackward)
  }

  /**
   * Load tensorflow checkpoints
   * @param graphFile
   * @param binFile
   * @tparam T
   * @return
   */
  def tensorflowCheckpoints[T: ClassTag](graphFile: String, binFile: String,
    byteOrder: ByteOrder = ByteOrder.LITTLE_ENDIAN)(implicit ev: TensorNumeric[T]): Session[T] = {
    TensorflowLoader.checkpoints(graphFile, binFile, byteOrder)
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
      require(parameters(i).isContiguous(), "parameters should be contiguous")
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

  def isCompact[@specialized(Float, Double) T: ClassTag](parameters: Array[Tensor[T]])(
    implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(parameters.length > 0,
      "The length of paramters should >= 0" +
      "parameter length" +
        s" ${parameters.length}")
    var i = 1
    val storage = parameters(0).storage()
    var length = parameters(0).nElement()
    val offset = parameters(0).storageOffset()
    // make sure parameters is shared and contiguous
    while (i < parameters.length) {
      if (!storage.eq(parameters(i).storage())) {
        return null
      }
      if (offset + length != parameters(i).storageOffset()) {
        return null
      }
      length += parameters(i).nElement()
      i += 1
    }

    Tensor(storage, offset, Array(length))
  }
}
