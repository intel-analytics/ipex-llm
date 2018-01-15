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

package com.intel.analytics.bigdl.nn.quantized

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.nn.tf.WithoutInput
import com.intel.analytics.bigdl.nn.{Cell, Container, Graph, Input, TimeDistributed, Linear => NNLinear, SpatialConvolution => NNConv, SpatialDilatedConvolution => NNDilatedConv}
import com.intel.analytics.bigdl.tensor.{QuantizedTensor, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Node
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

object Utils {
  type ModuleNode[R] = AbstractModule[Activity, Activity, R]
  type SeqNodes[R] = Seq[Node[ModuleNode[R]]]
  type ArrayNodes[R] = Array[Node[ModuleNode[R]]]
  type ANode[R] = Node[ModuleNode[R]]
  type AbsModule[R] = AbstractModule[Activity, Activity, R]

  /**
   * delete parameters of SpatialConvolution, SpatialDilatedConvolution) and linear.
   *
   * because it will make all parameters into a long array in a BigDL model by default,
   * so the origin parameters will exist in the quantized model. We have to delete them
   * for reducing the size.
   *
   * After deleting all these matched parameters, it will make a **new** long array of
   * other layers parameters.
   *
   * @param parameters parameters of all layers
   * @tparam T data type Float or Double
   * @return parameters reorganized
   */
  def reorganizeParameters[T: ClassTag](parameters: Array[Tensor[T]])(
    implicit ev: TensorNumeric[T]): Tensor[T] = {
    var length = 0
    for (i <- parameters.indices) {
      if (!parameters(i).isInstanceOf[QuantizedTensor[T]]) {
        length += parameters(i).nElement()
      }
    }

    val result = Tensor[T](length)

    var offset = 0
    for (i <- parameters.indices) {
      val parameter = parameters(i)

      if (!parameter.isInstanceOf[QuantizedTensor[T]]) {
        val length = parameter.nElement()

        val (src, srcOffset) = (parameter.storage().array(), parameter.storageOffset() - 1)
        val (dst, dstOffset) = (result.storage().array(), offset)

        val (size, stride) = (parameter.size(), parameter.stride())

        System.arraycopy(src, srcOffset, dst, dstOffset, length)
        parameter.set(result.storage(), offset + 1, size, stride)

        offset += length
      }
    }

    result
  }
}
