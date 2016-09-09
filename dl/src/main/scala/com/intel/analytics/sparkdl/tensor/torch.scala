/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.sparkdl.tensor

import java.io.Serializable

import breeze.linalg.{DenseMatrix => BrzDenseMatrix, DenseVector => BrzDenseVector}
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.utils.{File, TorchObject}
import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector}

import scala.reflect.ClassTag

// scalastyle:off
object torch {
  // scalastyle:on

  // scalastyle:off methodName
  def Tensor[@specialized(Float, Double) T: ClassTag]()(
    implicit ev: TensorNumeric[T]): Tensor[T] = new DenseTensor[T]()

  def Tensor[@specialized(Float, Double) T: ClassTag](d1: Int)(
    implicit ev: TensorNumeric[T]): Tensor[T] = new DenseTensor[T](d1)

  def Tensor[@specialized(Float, Double) T: ClassTag](d1: Int, d2: Int)(
    implicit ev: TensorNumeric[T]): Tensor[T] = new DenseTensor[T](d1, d2)

  def Tensor[@specialized(Float, Double) T: ClassTag](d1: Int, d2: Int, d3: Int)(
    implicit ev: TensorNumeric[T]): Tensor[T] = new DenseTensor[T](d1, d2, d3)

  def Tensor[@specialized(Float, Double) T: ClassTag](d1: Int, d2: Int, d3: Int, d4: Int)(
    implicit ev: TensorNumeric[T]): Tensor[T] = new DenseTensor[T](d1, d2, d3, d4)

  def Tensor[@specialized(Float, Double) T: ClassTag](d1: Int, d2: Int, d3: Int, d4: Int, d5: Int)(
    implicit ev: TensorNumeric[T]): Tensor[T] = new DenseTensor[T](d1, d2, d3, d4, d5)

  def Tensor[@specialized(Float, Double) T: ClassTag](dims: Int*)(
    implicit ev: TensorNumeric[T]): Tensor[T] =
    new DenseTensor[T](new ArrayStorage[T](new Array[T](dims.product)), 0, dims.toArray,
      DenseTensor.size2Stride(dims.toArray), dims.length)

  def Tensor[@specialized(Float, Double) T: ClassTag](sizes: Array[Int])(
    implicit ev: TensorNumeric[T]): Tensor[T] =
    new DenseTensor(new ArrayStorage[T](new Array[T](sizes.product)), 0, sizes.clone(),
      DenseTensor.size2Stride(sizes.clone()), sizes.length)

  def Tensor[@specialized(Float, Double) T: ClassTag](storage: Storage[T])(
    implicit ev: TensorNumeric[T]): Tensor[T] = {
    new DenseTensor(storage.asInstanceOf[Storage[T]])
  }

  def Tensor[@specialized(Float, Double) T: ClassTag](storage: Storage[T], storageOffset: Int,
    size: Array[Int] = null, stride: Array[Int] = null)
    (implicit ev: TensorNumeric[T]): Tensor[T] = {
    new DenseTensor(storage.asInstanceOf[Storage[T]], storageOffset, size, stride)
  }

  def Tensor[@specialized(Float, Double) T: ClassTag](other: Tensor[T])(
    implicit ev: TensorNumeric[T]): Tensor[T] = new DenseTensor(other)

  def Tensor[@specialized(Float, Double) T: ClassTag](vector: BrzDenseVector[T])(
    implicit ev: TensorNumeric[T]): Tensor[T] = torch.Tensor(torch.storage(vector.data),
    vector.offset + 1, Array(vector.length), Array(vector.stride))

  def Tensor(vector: DenseVector): Tensor[Double] =
    torch.Tensor[Double](torch.storage(vector.toArray))

  def Tensor[@specialized(Float, Double) T: ClassTag](matrix: BrzDenseMatrix[T])(
    implicit ev: TensorNumeric[T]): Tensor[T] = torch.Tensor(torch.storage(matrix.data),
    matrix.offset + 1, Array(matrix.rows, matrix.cols),
    if (matrix.isTranspose) Array(1, matrix.majorStride) else Array(matrix.majorStride, 1))

  def Tensor(matrix: DenseMatrix): Tensor[Double] = {
    val strides = if (matrix.isTransposed) {
      Array(matrix.numCols, 1)
    } else {
      Array(1, matrix.numRows) // column major
    }
    torch.Tensor(torch.storage(matrix.toArray), 1, Array(matrix.numRows, matrix.numCols), strides)
  }
  // scalastyle:on methodName

  def randperm[@specialized(Float, Double) T: ClassTag](size: Int)(
    implicit ev: TensorNumeric[T]): Tensor[T] = DenseTensor.randperm[T](size)

  def expand[T](tensor: Tensor[T], sizes: Int*): Tensor[T] = tensor.expand(sizes.toArray)

  def expandAs[T](tensor: Tensor[T], template: Tensor[T]): Tensor[T] = tensor.expandAs(template)

  def repeatTensor[T](tensor: Tensor[T], sizes: Int*): Tensor[T] =
    tensor.repeatTensor(sizes.toArray)

  def storage[T: ClassTag](): Storage[T] = new ArrayStorage[T](new Array[T](0))

  def storage[T: ClassTag](size: Int): Storage[T] = new ArrayStorage[T](new Array[T](size))

  def storage[@specialized(Float, Double) T: ClassTag](data: Array[T]): Storage[T] =
    new ArrayStorage[T](data)

  def load[T](fileName: String): T = File.load[T](fileName)

  def loadObj[T](fileName: String): T = File.loadObj[T](fileName)

  def save(data: Any, fileName: String, objectType: TorchObject): Unit =
    File.save(data, fileName, objectType)

  def saveObj(obj: Serializable, fileName: String, isOverwrite: Boolean = false): Unit =
    File.save(obj, fileName, isOverwrite)
}