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
package com.intel.analytics.bigdl.utils.tf.loaders

import java.nio.ByteOrder

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.tf.StrideSlice
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Node
import com.intel.analytics.bigdl.utils.tf.Context
import org.tensorflow.framework.NodeDef

import scala.reflect.ClassTag

class StridedSlice extends TensorflowOpsLoader {

  import Utils._

  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder,
    context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {

    new StridedSliceLoadTF[T]()
  }
}

class StridedSliceLoadTF[T: ClassTag]()(implicit ev: TensorNumeric[T])
  extends Adapter[T](Array(2, 3, 4)) {
  import StridedSlice._

  override def build(tensorArrays: Array[Tensor[_]]): AbstractModule[Activity, Activity, T] = {
    val start = oneDTensorToArray(tensorArrays(0).asInstanceOf[Tensor[Int]])
    val end = oneDTensorToArray(tensorArrays(1).asInstanceOf[Tensor[Int]])
    val stride = oneDTensorToArray(tensorArrays(2).asInstanceOf[Tensor[Int]])

    val specs = (start zip end zip stride).zipWithIndex
      .map(elem => (elem._2 + 1, elem._1._1._1 + 1, elem._1._1._2 + 1, elem._1._2))


    StrideSlice[T](specs)
  }
}

object StridedSlice {
  def oneDTensorToArray(tensor: Tensor[Int]): Array[Int] = {
    require(tensor.nDimension() == 1, "1D tensor required")
    val result = new Array[Int](tensor.nElement())
    var i = 0
    while(i < tensor.nElement()) {
      result(i) = tensor.valueAt(i + 1)
      i += 1
    }
    result
  }
}

