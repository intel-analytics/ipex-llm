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
import com.intel.analytics.bigdl.nn.tf.{StridedSlice => StridedSliceOps}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.tf.Context
import org.tensorflow.framework.{DataType, NodeDef}

import scala.reflect.ClassTag

class StridedSlice extends TensorflowOpsLoader {

  import Utils._

  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder,
    context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {

    val t = getType(nodeDef, "T")
    val beginMask = getInt(nodeDef.getAttrMap, "begin_mask")
    val ellipsisMask = getInt(nodeDef.getAttrMap, "ellipsis_mask")
    val endMask = getInt(nodeDef.getAttrMap, "end_mask")
    val newAxisMask = getInt(nodeDef.getAttrMap, "new_axis_mask")
    val shrinkAxisMask = getInt(nodeDef.getAttrMap, "shrink_axis_mask")

    if (t == DataType.DT_INT32) {
      StridedSliceOps[T, Int](beginMask, endMask, ellipsisMask,
        newAxisMask, shrinkAxisMask, true)
    } else if (t == DataType.DT_FLOAT) {
      StridedSliceOps[T, Float](beginMask, endMask, ellipsisMask,
        newAxisMask, shrinkAxisMask, true)
    } else if (t == DataType.DT_DOUBLE) {
      StridedSliceOps[T, Double](beginMask, endMask, ellipsisMask,
        newAxisMask, shrinkAxisMask, true)
    } else {
      throw new UnsupportedOperationException(s"Not support load StridedSlice with type ${t}")
    }
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

