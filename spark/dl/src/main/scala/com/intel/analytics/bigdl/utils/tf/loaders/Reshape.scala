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
import com.intel.analytics.bigdl.nn.{Reshape => ReshapeOps}
import com.intel.analytics.bigdl.nn.InferReshape
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.tf.Context
import org.tensorflow.framework.NodeDef

import scala.reflect.ClassTag

class Reshape extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder,
    context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    new ReshapeLoadTF[T]()
  }
}

class ReshapeLoadTF[T: ClassTag]()(implicit ev: TensorNumeric[T]) extends Adapter[T](Array(2)) {
  override def build(tensorArrays: Array[Tensor[_]]): AbstractModule[Activity, Activity, T] = {
    val sizes = tensorArrays(0).asInstanceOf[Tensor[Int]]


    val batchMode = if (sizes.nDimension() >= 1 && sizes.nElement() > 0) {
      sizes.valueAt(1) == -1
    } else {
      false
    }
    val arraySize = new Array[Int](if (batchMode) sizes.nElement() - 1 else sizes.nElement())
    var i = if (batchMode) 2 else 1
    var k = 0
    while(i <= sizes.nElement()) {
      arraySize(k) = sizes.valueAt(i)
      k += 1
      i += 1
    }
    val infer = arraySize.contains(-1)
    if (infer) InferReshape[T](size = arraySize, batchMode)
    else ReshapeOps[T](size = arraySize, Some(batchMode))
  }
}

