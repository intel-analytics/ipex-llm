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
import com.intel.analytics.bigdl.nn.ops.{Cast => CastOps}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.tf.Context
import org.tensorflow.framework.{DataType, NodeDef}

import scala.reflect.ClassTag

class Cast extends TensorflowOpsLoader {

  import Utils._

  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    val attr = nodeDef.getAttrMap
    val dataType = getType(attr, "DstT")

    val layer = dataType match {
      case DataType.DT_INT8 => CastOps[T, Int]()
      case DataType.DT_INT16 => CastOps[T, Int]()
      case DataType.DT_UINT8 => CastOps[T, Int]()
      case DataType.DT_UINT16 => CastOps[T, Int]()
      case DataType.DT_INT32 => CastOps[T, Int]()
      case DataType.DT_INT64 => CastOps[T, Int]()
      case DataType.DT_BOOL => CastOps[T, Boolean]()
      case DataType.DT_STRING => CastOps[T, String]()
      case DataType.DT_FLOAT => CastOps[T, Float]()
      case DataType.DT_DOUBLE => CastOps[T, Double]()
      case _ => throw new UnsupportedOperationException("Unsupported data type: "
        + dataType.toString)
    }
    layer
  }
}
