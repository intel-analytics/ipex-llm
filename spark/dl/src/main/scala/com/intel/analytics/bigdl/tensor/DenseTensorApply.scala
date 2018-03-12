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

package com.intel.analytics.bigdl.tensor

object DenseTensorApply {
  /**
   * Iterate through tensor1, and apply func to the elements,
   * set function result to tensor 2
   *
   * @param tensor1 the tensor1
   * @param tensor2 the result tensor
   * @param func    (tensor1Data, tensor1Offset, tensor2Data,
   *                tensor2Offset)
   */
  def apply1[A, B](tensor1: Tensor[A], tensor2: Tensor[B],
    func: TensorDiffTypeFunc4[A, B]): Unit = {

    if (tensor1.isEmpty) {
      return
    }

    // shortcut for scalar
    if (tensor1.isScalar && tensor2.isScalar) {
      val data1 = tensor1.storage().array()
      val index1 = tensor1.storageOffset() - 1
      val data2 = tensor2.storage().array()
      val index2 = tensor2.storageOffset() - 1
      func(data1, index1, data2, index2)
      return
    }

    val stride1 = getStride(tensor1)
    val stride2 = getStride(tensor2)
    val (largestDim1, largestSize1) = getLargestContiguousSize(tensor1)
    val (largestDim2, largestSize2) = getLargestContiguousSize(tensor2)
    val counter1 = getCounter(largestDim1)
    val counter2 = getCounter(largestDim2)
    val data1 = tensor1.storage().array()
    val data2 = tensor2.storage().array()
    var offset1 = tensor1.storageOffset() - 1
    var offset2 = tensor2.storageOffset() - 1
    var hasFinished1 = false
    var hasFinished2 = false
    var i1 = 0
    var i2 = 0
    while (!hasFinished1 && !hasFinished2) {
      while (i1 < largestSize1 && i2 < largestSize2) {
        val index1 = offset1 + i1 * stride1
        val index2 = offset2 + i2 * stride2
        func(data1, index1, data2, index2)
        i1 += 1
        i2 += 1
      }
      val r1 = updateCounter(tensor1, counter1, offset1, largestDim1)
      val r2 = updateCounter(tensor2, counter2, offset2, largestDim2)
      hasFinished1 = r1._1
      hasFinished2 = r2._1
      offset1 = r1._2
      offset2 = r2._2
      i1 = 0
      i2 = 0
    }
  }

  /**
   * Iterate through tensor1, tensor2, and apply func to the elements
   *
   * @param tensor
   * @param func (tensor1Data, tensor1Offset)
   */
  def apply1[@specialized(Float, Double) T](
    tensor: Tensor[T], func: TensorFunc2[T]): Unit = {

    if (tensor.isEmpty) {
      return
    }

    // shortcut for scalar
    if (tensor.isScalar) {
      val data = tensor.storage().array()
      val index = tensor.storageOffset() - 1
      func(data, index)
      return
    }


    val stride = getStride(tensor)
    val (largestDim, largestSize) = getLargestContiguousSize(tensor)
    val counter = getCounter(largestDim)
    val data = tensor.storage().array()
    var offset = tensor.storageOffset() - 1
    var hasFinished = false
    var i = 0
    while (!hasFinished) {
      while (i < largestSize) {
        val index = offset + i * stride
        func(data, index)
        i += 1
      }
      val r = updateCounter(tensor, counter, offset, largestDim)
      hasFinished = r._1
      offset = r._2
      i = 0
    }
  }

  /**
   * Iterate through tensor1, tensor2, and apply func to the elements
   *
   * @param tensor1 the tensor
   * @param tensor2 the tensor
   * @param func    (tensor1Data, tensor1Offset, tensor2Data, tensor2Offset)
   */
  def apply2[T](tensor1: Tensor[T], tensor2: Tensor[T],
    func: TensorFunc4[T]): Unit = {
    require(tensor1.nElement() == tensor2.nElement(),
      s"inconsistent tensor size: ${tensor1.nElement()} == ${tensor2.nElement()}")

    if (tensor1.isEmpty) {
      return
    }

    // shortcut for scalar
    if (tensor1.isScalar && tensor2.isScalar) {
      val tensor1Data = tensor1.storage().array()
      val tensor2Data = tensor2.storage().array()
      val tensor1Index = tensor1.storageOffset() - 1
      val tensor2Index = tensor2.storageOffset() - 1
      func(tensor1Data, tensor1Index, tensor2Data, tensor2Index)
      return
    }

    val tensor1Data = tensor1.storage().array()
    var tensor1Offset = tensor1.storageOffset() - 1
    val tensor2Data = tensor2.storage().array()
    var tensor2Offset = tensor2.storageOffset() - 1

    var adjacent = false
    if (tensor1.nDimension == 1 && tensor2.nDimension == 1 && tensor1.stride(1) == 1 &&
      tensor2.stride(1) == 1) {
      adjacent = true
    }
    if (tensor1.nDimension == 2 && tensor2.nDimension == 2) {
      if (tensor1.stride(2) == 1 && tensor2.stride(2) == 1 && tensor1.stride(1) == tensor1.size(2)
        && tensor2.stride(1) == tensor2.size(2)) {
        adjacent = true
      }

      if (tensor1.stride(1) == 1 && tensor2.stride(1) == 1 && tensor1.stride(2) == tensor1.size(1)
        && tensor2.stride(2) == tensor2.size(1)) {
        adjacent = true
      }
    }
    if (adjacent) {
      var i = 0
      while (i < tensor1.nElement()) {
        func(tensor1Data, tensor1Offset + i, tensor2Data, tensor2Offset + i)
        i += 1
      }
      return
    }

    val tensor1Stride = getStride(tensor1)
    val (largestDim1, largestSize1) = getLargestContiguousSize(tensor1)
    val counter1 = getCounter(largestDim1)
    val tensor2Stride = getStride(tensor2)
    val (largestDim2, largestSize2) = getLargestContiguousSize(tensor2)
    val counter2 = getCounter(largestDim2)

    var hasFinished = false
    var i1 = 0
    var i2 = 0
    while (!hasFinished) {
      while (i1 < largestSize1 && i2 < largestSize2) {
        func(tensor1Data, tensor1Offset + i1 * tensor1Stride, tensor2Data,
          tensor2Offset + i2 * tensor2Stride)
        i1 = i1 + 1
        i2 = i2 + 1
      }

      if (i1 == largestSize1) {
        val r = updateCounter(tensor1, counter1, tensor1Offset, largestDim1)
        hasFinished = r._1
        tensor1Offset = r._2
        i1 = 0
      }

      if (i2 == largestSize2) {
        val r = updateCounter(tensor2, counter2, tensor2Offset, largestDim2)
        hasFinished = r._1
        tensor2Offset = r._2
        i2 = 0
      }
    }
  }

  /**
   * Iterate through tensor1, tensor2, tensor3, and apply func to the elements
   *
   * @param tensor1 the tensor
   * @param tensor2 the tensor
   * @param tensor3 the tensor
   * @param func    (tensor1Data, tensor1Offset, tensor2Data, tensor2Offset, tensor3Data,
   *                tensor3Offset)
   */
  private[bigdl] def apply3[@specialized(Float, Double) T](tensor1: Tensor[T],
    tensor2: Tensor[T], tensor3: Tensor[T],
    func: TensorFunc6[T]): Unit = {

    require(tensor1.nElement() == tensor2.nElement() && tensor2.nElement() == tensor3.nElement(),
      "inconsistent tensor size")

    if (tensor1.isEmpty) {
      return
    }

    // shortcut for scalar
    if (tensor1.isScalar && tensor2.isScalar && tensor3.isScalar) {
      val tensor1Data = tensor1.storage().array()
      val tensor2Data = tensor2.storage().array()
      val tensor3Data = tensor3.storage().array()
      val tensor1Index = tensor1.storageOffset() - 1
      val tensor2Index = tensor2.storageOffset() - 1
      val tensor3Index = tensor3.storageOffset() - 1
      func(tensor1Data, tensor1Index, tensor2Data, tensor2Index, tensor3Data, tensor3Index)
      return
    }

    val tensor1Data = tensor1.storage().array()
    var tensor1Offset = tensor1.storageOffset() - 1
    val tensor1Stride = getStride(tensor1)
    val (tensor1Dim, tensor1Size) = getLargestContiguousSize(tensor1)
    val tensor1Counter = getCounter(tensor1Dim)

    val tensor2Data = tensor2.storage().array()
    var tensor2Offset = tensor2.storageOffset() - 1
    val tensor2Stride = getStride(tensor2)
    val (tensor2Dim, tensor2Size) = getLargestContiguousSize(tensor2)
    val tensor2Counter = getCounter(tensor2Dim)

    val tensor3Data = tensor3.storage().array()
    var tensor3Offset = tensor3.storageOffset() - 1
    val tensor3Stride = getStride(tensor3)
    val (tensor3Dim, tensor3Size) = getLargestContiguousSize(tensor3)
    val tensor3Counter = getCounter(tensor3Dim)

    var hasFinished = false
    var i1 = 0
    var i2 = 0
    var i3 = 0
    while (!hasFinished) {
      while (i1 < tensor1Size && i2 < tensor2Size && i3 < tensor3Size) {
        func(tensor1Data, tensor1Offset + i1 * tensor1Stride, tensor2Data,
          tensor2Offset + i2 * tensor2Stride,
          tensor3Data, tensor3Offset + i3 * tensor3Stride)
        i1 += 1
        i2 += 1
        i3 += 1
      }

      if (i1 == tensor1Size) {
        val r = updateCounter(tensor1, tensor1Counter, tensor1Offset, tensor1Dim)
        hasFinished = r._1
        tensor1Offset = r._2
        i1 = 0
      }

      if (i2 == tensor2Size) {
        val r = updateCounter(tensor2, tensor2Counter, tensor2Offset, tensor2Dim)
        hasFinished = r._1
        tensor2Offset = r._2
        i2 = 0
      }

      if (i3 == tensor3Size) {
        val r = updateCounter(tensor3, tensor3Counter, tensor3Offset, tensor3Dim)
        hasFinished = r._1
        tensor3Offset = r._2
        i3 = 0
      }
    }
  }

  /**
   * Get the stride discard dimensions with size 1
   *
   * @param tensor tensor
   * @return
   */
  def getStride[T](tensor: Tensor[T]): Int = {
    var d = tensor.nDimension()
    while (d > 0) {
      if (tensor.size(d) != 1) {
        return tensor.stride(d)
      }
      d -= 1
    }

    0
  }

  def getLargestContiguousSize[T](tensor: Tensor[T]): (Int, Int) = {
    var largestSize = 1
    var largestDim = tensor.nDimension()
    while (largestDim > 0) {
      if (tensor.size(largestDim) != 1) {
        if (tensor.stride(largestDim) == largestSize) {
          largestSize = largestSize * tensor.size(largestDim)
        } else {
          return (largestDim, largestSize)
        }
      }
      largestDim -= 1
    }
    (largestDim, largestSize)
  }

  def getCounter(largestDim: Int): Array[Int] = {
    val counter = new Array[Int](largestDim)
    var d = 0
    while (d < largestDim) {
      counter(d) = 0
      d += 1
    }
    counter
  }

  def updateCounter[T](tensor: Tensor[T], counter: Array[Int], offset: Int,
    dim: Int): (Boolean, Int) = {
    if (dim == 0) {
      return (true, offset)
    }

    var _offset = offset
    var i = dim
    while (i > 0) {
      counter(i - 1) += 1
      _offset += tensor.stride(i)
      if (counter(i - 1) == tensor.size(i)) {
        if (i == 1) {
          return (true, _offset)
        } else {
          _offset -= counter(i - 1) * tensor.stride(i)
          counter(i - 1) = 0
        }
      } else {
        return (false, _offset)
      }
      i -= 1
    }

    (false, _offset)
  }

  /**
   * Iterate through tensor1, tensor2, and apply func to the elements,
   * set function result to tensor 3
   *
   * @param tensor1 the tensor1
   * @param tensor2 the tensor2
   * @param tensor3 the result tensor
   * @param func    (tensor1Data, tensor1Offset, tensor2Data,
   *                tensor2Offset, tensor3Data, tensor3Offset)
   */
  def apply2[A, B, C](tensor1: Tensor[A], tensor2: Tensor[B], tensor3: Tensor[C],
    func: TensorDiffTypeFunc6[A, B, C])
  : Unit = {
    require(tensor1.nElement() == tensor2.nElement(),
      s"inconsistent tensor size: ${tensor1.nElement()} == ${tensor2.nElement()}")

    if (tensor1.isEmpty) {
      return
    }

    // shortcut for scalar
    if (tensor1.isScalar && tensor2.isScalar) {
      val tensor1Data = tensor1.storage().array()
      val tensor2Data = tensor2.storage().array()
      val tensor3Data = tensor3.storage().array()
      val tensor1Index = tensor1.storageOffset() - 1
      val tensor2Index = tensor2.storageOffset() - 1
      val tensor3Index = tensor3.storageOffset() - 1
      func(tensor1Data, tensor1Index, tensor2Data, tensor2Index, tensor3Data, tensor3Index)
      return
    }

    val tensor1Data = tensor1.storage().array()
    var tensor1Offset = tensor1.storageOffset() - 1
    val tensor2Data = tensor2.storage().array()
    var tensor2Offset = tensor2.storageOffset() - 1
    val tensor3Data = tensor3.storage().array()
    val tensor3Offset = tensor3.storageOffset() - 1

    var adjacent = false
    if (tensor1.nDimension == 1 && tensor2.nDimension == 1 && tensor1.stride(1) == 1 &&
      tensor2.stride(1) == 1) {
      adjacent = true
    }
    if (tensor1.nDimension == 2 && tensor2.nDimension == 2) {
      if (tensor1.stride(2) == 1 && tensor2.stride(2) == 1 && tensor1.stride(1) == tensor1.size(2)
        && tensor2.stride(1) == tensor2.size(2)) {
        adjacent = true
      }

      if (tensor1.stride(1) == 1 && tensor2.stride(1) == 1 && tensor1.stride(2) == tensor1.size(1)
        && tensor2.stride(2) == tensor2.size(1)) {
        adjacent = true
      }
    }
    if (adjacent) {
      var i = 0
      while (i < tensor1.nElement()) {
        func(
          tensor1Data, tensor1Offset + i,
          tensor2Data, tensor2Offset + i,
          tensor3Data, tensor3Offset + i)
        i += 1
      }
      return
    }

    val tensor1Stride = getStride(tensor1)
    val (largestDim1, largestSize1) = getLargestContiguousSize(tensor1)
    val counter1 = getCounter(largestDim1)
    val tensor2Stride = getStride(tensor2)
    val (largestDim2, largestSize2) = getLargestContiguousSize(tensor2)
    val counter2 = getCounter(largestDim2)
    val tensor3Stride = getStride(tensor3)
    val (largestDim3, largestSize3) = getLargestContiguousSize(tensor3)
    val counter3 = getCounter(largestDim3)

    var hasFinished = false
    var i1 = 0
    var i2 = 0
    var i3 = 0
    while (!hasFinished) {
      while (i1 < largestSize1 && i2 < largestSize2) {
        func(
          tensor1Data, tensor1Offset + i1 * tensor1Stride,
          tensor2Data, tensor2Offset + i2 * tensor2Stride,
          tensor3Data, tensor3Offset + i3 * tensor3Stride
        )
        i1 = i1 + 1
        i2 = i2 + 1
        i3 = i3 + 1
      }

      if (i1 == largestSize1) {
        val r = updateCounter(tensor1, counter1, tensor1Offset, largestDim1)
        hasFinished = r._1
        tensor1Offset = r._2
        i1 = 0
      }

      if (i2 == largestSize2) {
        val r = updateCounter(tensor2, counter2, tensor2Offset, largestDim2)
        hasFinished = r._1
        tensor2Offset = r._2
        i2 = 0
      }

      if (i3 == largestSize3) {
        val r = updateCounter(tensor3, counter3, tensor3Offset, largestDim3)
        hasFinished = r._1
        tensor2Offset = r._2
        i3 = 0
      }
    }
  }

}
