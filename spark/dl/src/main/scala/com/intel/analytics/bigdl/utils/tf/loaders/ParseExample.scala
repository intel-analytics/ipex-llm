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
import com.intel.analytics.bigdl.nn.ops.{ParseExample => ParseExampleOperation}
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.tf.Context
import org.tensorflow.framework.{DataType, NodeDef}

import collection.JavaConverters._
import scala.reflect.ClassTag

class ParseExample extends TensorflowOpsLoader {

  import Utils._

  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder,
    context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    val Ndense = nodeDef.getAttrMap.get("Ndense").getI.toInt
    val Tdense = nodeDef.getAttrMap.get("Tdense")
      .getList.getTypeList.asScala
      .map {
        case DataType.DT_INT64 => LongType
        case DataType.DT_INT32 => IntType
        case DataType.DT_FLOAT => FloatType
        case DataType.DT_DOUBLE => DoubleType
        case DataType.DT_STRING => StringType
      }
    val denseShapes = nodeDef.getAttrMap.get("dense_shapes")
      .getList.getShapeList.asScala
      .map { shapeProto =>
        shapeProto.getDimList.asScala.map(_.getSize.toInt).toArray
      }

    new ParseExampleOperation[T](Ndense, Tdense, denseShapes)
  }
}

