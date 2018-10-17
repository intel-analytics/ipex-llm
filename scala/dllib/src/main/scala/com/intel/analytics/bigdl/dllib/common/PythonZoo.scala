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
package com.intel.analytics.zoo.common

import com.intel.analytics.bigdl.python.api.{JTensor, PythonBigDLKeras}
import com.intel.analytics.bigdl.tensor.{DenseType, SparseType, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

object PythonZoo {

  def ofFloat(): PythonZoo[Float] = new PythonZoo[Float]()

  def ofDouble(): PythonZoo[Double] = new PythonZoo[Double]()

}


class PythonZoo[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonBigDLKeras[T] {

  private val typeName = {
    val cls = implicitly[ClassTag[T]].runtimeClass
    cls.getSimpleName
  }

  override def toTensor(jTensor: JTensor): Tensor[T] = {
    if (jTensor == null) return null

    this.typeName match {
      case "float" =>
        if (null == jTensor.indices) {
          if (jTensor.shape == null || jTensor.shape.product == 0) {
            Tensor()
          } else {
            Tensor(jTensor.storage.map(x => ev.fromType(x)), jTensor.shape)
          }
        } else {
          Tensor.sparse(jTensor.indices, jTensor.storage.map(x => ev.fromType(x)), jTensor.shape)
        }
      case "double" =>
        if (null == jTensor.indices) {
          if (jTensor.shape == null || jTensor.shape.product == 0) {
            Tensor()
          } else {
            Tensor(jTensor.storage.map(x => ev.fromType(x.toDouble)), jTensor.shape)
          }
        } else {
          Tensor.sparse(jTensor.indices,
            jTensor.storage.map(x => ev.fromType(x.toDouble)), jTensor.shape)
        }
      case t: String =>
        throw new IllegalArgumentException(s"Not supported type: ${t}")
    }
  }

  override def toJTensor(tensor: Tensor[T]): JTensor = {
    // clone here in case the the size of storage larger then the size of tensor.
    require(tensor != null, "tensor cannot be null")
    tensor.getTensorType match {
      case SparseType =>
        // Note: as SparseTensor's indices is inaccessible here,
        // so we will transfer it to DenseTensor. Just for testing.
        if (tensor.nElement() == 0) {
          JTensor(Array(), Array(0), bigdlType = typeName)
        } else {
          val cloneTensor = Tensor.dense(tensor)
          val result = JTensor(cloneTensor.storage().array().map(i => ev.toType[Float](i)),
            cloneTensor.size(), bigdlType = typeName)
          result
        }
      case DenseType =>
        if (tensor.nElement() == 0) {
          if (tensor.dim() == 0) {
            JTensor(null, null, bigdlType = typeName)
          } else {
            JTensor(Array(), tensor.size(), bigdlType = typeName)
          }
        } else {
          val cloneTensor = tensor.clone()
          val result = JTensor(cloneTensor.storage().array().map(i => ev.toType[Float](i)),
            cloneTensor.size(), bigdlType = typeName)
          result
        }
      case _ =>
        throw new IllegalArgumentException(s"toJTensor: Unsupported tensor type" +
          s" ${tensor.getTensorType}")
    }
  }

}
