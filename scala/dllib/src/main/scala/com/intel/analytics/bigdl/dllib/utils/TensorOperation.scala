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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

object TensorOperation {

  def expandSize[T: ClassTag](tensor: Tensor[T], other: Tensor[T]): Array[Int] = {
    val errorMsg = s"tensor size not match ${tensor.size.mkString("x")} " +
      s"${other.size.mkString("x")}"
    val longTensor = if (tensor.dim() > other.dim()) tensor else other
    val shortTensor = if (tensor.dim() > other.dim()) other else tensor
    val ndim = longTensor.nDimension()
    val delta = longTensor.nDimension() - shortTensor.nDimension()
    val size = new Array[Int](ndim)
    var i = ndim - 1
    while (i >= delta) {
      require(longTensor.size(i + 1) == shortTensor.size(i + 1 - delta) ||
        longTensor.size(i + 1) == 1 ||
        shortTensor.size(i + 1 - delta) == 1, errorMsg)
      size(i) = math.max(longTensor.size(i + 1), shortTensor.size(i + 1 - delta))
      i -= 1
    }

    while (i >= 0) {
      size(i) = longTensor.size(i + 1)
      i -= 1
    }

    size
  }

  def expandTensor[T: ClassTag](tensor: Tensor[T], tensor2: Tensor[T])
                               (implicit ev: TensorNumeric[T]): Tensor[T] = {
    val targetSize = expandSize(tensor, tensor2)
    val expandStrides = new Array[Int](targetSize.length)

    val expandStridesX = new Array[Int](targetSize.length)
    var i = targetSize.length - 1
    val delta2 = targetSize.length - tensor2.nDimension
    while(i >= delta2) {
      if (tensor2.size(i + 1- delta2) != 1) expandStridesX(i) = tensor2.stride(i + 1- delta2)
      i -= 1
    }
    val expandX = Tensor[T](
      tensor2.storage(),
      tensor2.storageOffset(),
      targetSize,
      expandStridesX
    )
    if (targetSize.product != tensor.nElement()) {
      i = targetSize.length - 1
      val delta1 = targetSize.length - tensor.nDimension
      while (i >= delta1) {
        if (tensor.size(i + 1 - delta1) != 1) expandStrides(i) = tensor.stride(i + 1 - delta1)
        i -= 1
      }
      val tensor1 = Tensor[T](
        tensor.storage,
        tensor.storageOffset(),
        targetSize,
        expandStrides
      )
      val newTensor = Tensor[T]().resize(targetSize).add(tensor1)
      tensor.set(newTensor)
    }
    expandX
  }

  def subTensor[T: ClassTag](tensor: Tensor[T], tensor2: Tensor[T])
    (implicit ev: TensorNumeric[T]): Tensor[T] = {
    val expandedTensor = expandTensor(tensor, tensor2).contiguous()
    tensor.sub(expandedTensor)
    tensor
  }

  def divTensor[T: ClassTag](tensor: Tensor[T], tensor2: Tensor[T])
    (implicit ev: TensorNumeric[T]): Tensor[T] = {
    val expandedTensor = expandTensor(tensor, tensor2).contiguous()
    tensor.div(expandedTensor)
    tensor
  }
}
