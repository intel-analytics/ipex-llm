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
import com.intel.analytics.bigdl.nn.ops.{Sum => SumOps}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.tf.Context
import org.tensorflow.framework.{DataType, NodeDef}

import scala.reflect.ClassTag

class Sum extends TensorflowOpsLoader {

  import Utils._

  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    val attr = nodeDef.getAttrMap
    val keepDims = getBoolean(attr, "keep_dims")
    val dataType = getType(attr, "T")
    dataType match {
      case DataType.DT_INT8 =>
        SumOps[T, Int](keepDims, startFromZero = true)
      case DataType.DT_INT16 =>
        SumOps[T, Int](keepDims, startFromZero = true)
      case DataType.DT_UINT8 =>
        SumOps[T, Int](keepDims, startFromZero = true)
      case DataType.DT_UINT16 =>
        SumOps[T, Int](keepDims, startFromZero = true)
      case DataType.DT_INT32 =>
        SumOps[T, Int](keepDims, startFromZero = true)
      case DataType.DT_INT64 =>
        SumOps[T, Int](keepDims, startFromZero = true)
      case DataType.DT_FLOAT =>
        SumOps[T, Float](keepDims, startFromZero = true)
      case DataType.DT_DOUBLE =>
        SumOps[T, Double](keepDims, startFromZero = true)
      case _ => throw new UnsupportedOperationException()
    }
  }
}
