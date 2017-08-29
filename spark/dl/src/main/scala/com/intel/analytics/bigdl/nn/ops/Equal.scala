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
package com.intel.analytics.bigdl.nn.ops

import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class Equal[T: ClassTag]()
  (implicit ev: TensorNumeric[T]) extends Operation[Table, T] {

  /**
   * Iterate through tensor1, tensor2, and apply func to the elements
   *
   * @param tensor1 the tensor
   * @param tensor2 the tensor
   * @param func    (tensor1Data, tensor1Offset, tensor2Data, tensor2Offset)
   */
  def apply2[A, B, C](tensor1: Tensor[A], tensor2: Tensor[B], tensor3: Tensor[C],
    func: (Array[A], Int, Array[B], Int, Array[C], Int) => Unit)
  : Unit = {
    import com.intel.analytics.bigdl.tensor.DenseTensorApply._

    require(tensor1.nElement() == tensor2.nElement(),
      s"inconsistent tensor size: ${tensor1.nElement()} == ${tensor2.nElement()}")

    if (tensor1.nDimension == 0) {
      return
    }

    val tensor1Data = tensor1.storage().array()
    var tensor1Offset = tensor1.storageOffset() - 1
    val tensor2Data = tensor2.storage().array()
    var tensor2Offset = tensor2.storageOffset() - 1
    val tensor3Data = tensor3.storage().array()
    var tensor3Offset = tensor3.storageOffset() - 1

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
    }
  }

  def zipWith[A: ClassTag, B: ClassTag, C: ClassTag](
    t1: Tensor[A],
    t2: Tensor[B],
    t3: Tensor[C],
    func: (A, B) => C): Tensor[C] = {
    def func2(
      data1: Array[A], index1: Int,
      data2: Array[B], index2: Int,
      data3: Array[C], index3: Int): Unit = {
        data3(index3) = func(data1(index1), data2(index2))
    }
    apply2(t1, t2, t3, func2)
    t3
  }

  override def updateOutput(input: Table): Tensor[T] = {
    output.resizeAs(input(1))
    input[Tensor[_]](1) match {
      case t1 if t1.getType() == FloatType =>
        zipWith[Float, Float, Boolean](input[Tensor[Float]](1),
          input[Tensor[Float]](2), output.asInstanceOf[Tensor[Boolean]], (a, b) => a == b)
      case t2: Tensor[Boolean] =>
        zipWith[Boolean, Boolean, Boolean](input[Tensor[Boolean]](1),
          input[Tensor[Boolean]](2), output.asInstanceOf[Tensor[Boolean]], (a, b) => a == b)
      case t3: Tensor[Double] =>
        zipWith[Double, Double, Boolean](input[Tensor[Double]](1),
          input[Tensor[Double]](2), output.asInstanceOf[Tensor[Boolean]], (a, b) => a == b)
      case t4: Tensor[Char] =>
        zipWith[Char, Char, Boolean](input[Tensor[Char]](1),
          input[Tensor[Char]](2), output.asInstanceOf[Tensor[Boolean]], (a, b) => a == b)
      case t5: Tensor[String] =>
        zipWith[String, String, Boolean](input[Tensor[String]](1),
          input[Tensor[String]](2), output.asInstanceOf[Tensor[Boolean]], (a, b) => a == b)
      case t6: Tensor[Long] =>
        zipWith[Long, Long, Boolean](input[Tensor[Long]](1),
          input[Tensor[Long]](2), output.asInstanceOf[Tensor[Boolean]], (a, b) => a == b)
      case t7: Tensor[Short] =>
        zipWith[Short, Short, Boolean](input[Tensor[Short]](1),
          input[Tensor[Short]](2), output.asInstanceOf[Tensor[Boolean]], (a, b) => a == b)
      case t8: Tensor[Int] =>
        zipWith[Int, Int, Boolean](input[Tensor[Int]](1),
          input[Tensor[Int]](2), output.asInstanceOf[Tensor[Boolean]], (a, b) => a == b)
      case _ => throw new RuntimeException("Unsupported tensor type")
    }

    output
  }
}

object Equal {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): Operation[Table, T]
  = ModuleToOperation[Table, T](new Equal())
}