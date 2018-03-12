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
import com.intel.analytics.bigdl.nn.tf.{MergeOps, SwitchOps, Enter => EnterOps, Exit => ExitOps,
  LoopCondition => LoopConditionOps, NextIteration => NextIterationOps}
import com.intel.analytics.bigdl.nn.tf.ControlDependency
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.tf.Context
import com.intel.analytics.bigdl.utils.tf.Tensorflow._
import com.intel.analytics.bigdl.utils.tf.loaders.Utils.getType
import org.tensorflow.framework.{DataType, NodeDef}

import scala.reflect.ClassTag

private[bigdl] class Switch extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    new SwitchOps[T]()
  }
}

private[bigdl] class Exit extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    new ExitOps[T]()
  }
}

private[bigdl] class NextIteration extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    val t = getType(nodeDef.getAttrMap, "T")
    if (t == DataType.DT_FLOAT) {
      new NextIterationOps[T, Float]()
    } else if (t == DataType.DT_INT32) {
      new NextIterationOps[T, Int]()
    } else {
      throw new UnsupportedOperationException(s"Not support numeric type $t")
    }
  }
}

private[bigdl] class Enter extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    val frameName = stringAttr(nodeDef, "frame_name")
    new EnterOps[T](frameName)
  }
}

private[bigdl] class RefEnter extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    val frameName = stringAttr(nodeDef, "frame_name")
    new EnterOps[T](frameName)
  }
}

private[bigdl] class LoopCond extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    new LoopConditionOps[T]()
  }
}

private[bigdl] class Merge extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    new MergeOps[T]()
  }
}

private[bigdl] class ControlTrigger extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    new ControlDependency[T]()
  }
}

