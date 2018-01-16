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

import com.intel.analytics.bigdl.nn.Identity
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.ops._
import com.intel.analytics.bigdl.tensor.TensorNumericMath
import com.intel.analytics.bigdl.utils.tf.Context
import com.intel.analytics.bigdl.utils.tf.loaders.Utils._
import org.tensorflow.framework.{DataType, NodeDef}

import scala.reflect.ClassTag

class TensorArrayV3 extends TensorflowOpsLoader {

  override def build[T: ClassManifest](
    nodeDef: NodeDef,
    byteOrder: ByteOrder,
    context: Context[T]
  )(implicit ev: TensorNumericMath.TensorNumeric[T]): Module[T] = {
    val dynamicSize = getBoolean(nodeDef, "dynamic_size")
    val clearAfterRead = getBoolean(nodeDef, "clear_after_read")
    val identicalElementShapes = if (nodeDef.containsAttr("identical_element_shapes")) {
      getBoolean(nodeDef, "identical_element_shapes")
    } else {
      false
    }
    val tensorArrayName = getString(nodeDef, "tensor_array_name")

    val t = getType(nodeDef, "dtype")
    if (t == DataType.DT_FLOAT) {
      new TensorArrayCreator[T, Float](
        dynamicSize = dynamicSize,
        clearAfterRead = clearAfterRead,
        identicalElementShapes = identicalElementShapes,
        tensorArrayName = if (tensorArrayName == "") null else tensorArrayName
      )
    } else if (t == DataType.DT_DOUBLE) {
      new TensorArrayCreator[T, Double](
        dynamicSize = dynamicSize,
        clearAfterRead = clearAfterRead,
        identicalElementShapes = identicalElementShapes,
        tensorArrayName = if (tensorArrayName == "") null else tensorArrayName
      )
    } else if (t == DataType.DT_INT32) {
      new TensorArrayCreator[T, Int](
        dynamicSize = dynamicSize,
        clearAfterRead = clearAfterRead,
        identicalElementShapes = identicalElementShapes,
        tensorArrayName = if (tensorArrayName == "") null else tensorArrayName
      )
    } else {
      throw new UnsupportedOperationException(s"Not support load TensorArrayV3 with data type $t")
    }
  }
}

class TensorArrayGradV3 extends TensorflowOpsLoader {

  override def build[T: ClassManifest](
    nodeDef: NodeDef,
    byteOrder: ByteOrder,
    context: Context[T]
  )(implicit ev: TensorNumericMath.TensorNumeric[T]): Module[T] = {
    Identity()
  }
}

class TensorArrayGatherV3 extends TensorflowOpsLoader {
  override def build[T: ClassManifest](
    nodeDef: NodeDef,
    byteOrder: ByteOrder,
    context: Context[T]
  )(implicit ev: TensorNumericMath.TensorNumeric[T]): Module[T] = {
    val t = getType(nodeDef, "dtype")
    if (t == DataType.DT_FLOAT) {
      new TensorArrayGather[T, Float]()
    } else if (t == DataType.DT_DOUBLE) {
      new TensorArrayGather[T, Float]()
    } else if (t == DataType.DT_INT32) {
      new TensorArrayGather[T, Int]()
    } else {
      throw new UnsupportedOperationException(
        s"Not support load TensorArrayGatherV3 with data type $t")
    }
  }
}

class TensorArrayScatterV3 extends TensorflowOpsLoader {
  override def build[T: ClassManifest](
    nodeDef: NodeDef,
    byteOrder: ByteOrder,
    context: Context[T]
  )(implicit ev: TensorNumericMath.TensorNumeric[T]): Module[T] = {
    val t = getType(nodeDef, "T")
    if (t == DataType.DT_FLOAT) {
      new TensorArrayScatter[T, Float]()
    } else if (t == DataType.DT_DOUBLE) {
      new TensorArrayScatter[T, Float]()
    } else if (t == DataType.DT_INT32) {
      new TensorArrayScatter[T, Int]()
    } else {
      throw new UnsupportedOperationException(
        s"Not support load TensorArrayScatterV3 with data type $t")
    }
  }
}

class TensorArrayConcatV3 extends TensorflowOpsLoader {
  override def build[T: ClassManifest](
    nodeDef: NodeDef,
    byteOrder: ByteOrder,
    context: Context[T]
  )(implicit ev: TensorNumericMath.TensorNumeric[T]): Module[T] = {
    val t = getType(nodeDef, "dtype")
    if (t == DataType.DT_FLOAT) {
      new TensorArrayConcat[T, Float]()
    } else if (t == DataType.DT_DOUBLE) {
      new TensorArrayConcat[T, Float]()
    } else if (t == DataType.DT_INT32) {
      new TensorArrayConcat[T, Int]()
    } else {
      throw new UnsupportedOperationException(
        s"Not support load TensorArrayConcatV3 with data type $t")
    }
  }
}

class TensorArraySplitV3 extends TensorflowOpsLoader {
  override def build[T: ClassManifest](
    nodeDef: NodeDef,
    byteOrder: ByteOrder,
    context: Context[T]
  )(implicit ev: TensorNumericMath.TensorNumeric[T]): Module[T] = {
    val t = getType(nodeDef, "T")
    if (t == DataType.DT_FLOAT) {
      new TensorArraySplit[T, Float]()
    } else if (t == DataType.DT_DOUBLE) {
      new TensorArraySplit[T, Float]()
    } else if (t == DataType.DT_INT32) {
      new TensorArraySplit[T, Int]()
    } else {
      throw new UnsupportedOperationException(
        s"Not support load TensorArraySplitV3 with data type $t")
    }
  }
}

class TensorArrayReadV3 extends TensorflowOpsLoader {
  override def build[T: ClassManifest](
    nodeDef: NodeDef,
    byteOrder: ByteOrder,
    context: Context[T]
  )(implicit ev: TensorNumericMath.TensorNumeric[T]): Module[T] = {
    val t = getType(nodeDef, "dtype")
    if (t == DataType.DT_FLOAT) {
      new TensorArrayRead[T, Float]()
    } else if (t == DataType.DT_DOUBLE) {
      new TensorArrayRead[T, Float]()
    } else if (t == DataType.DT_INT32) {
      new TensorArrayRead[T, Int]()
    } else {
      throw new UnsupportedOperationException(
        s"Not support load TensorArrayReadV3 with data type $t")
    }
  }
}

class TensorArrayWriteV3 extends TensorflowOpsLoader {
  override def build[T: ClassManifest](
    nodeDef: NodeDef,
    byteOrder: ByteOrder,
    context: Context[T]
  )(implicit ev: TensorNumericMath.TensorNumeric[T]): Module[T] = {
    val t = getType(nodeDef, "T")
    if (t == DataType.DT_FLOAT) {
      new TensorArrayWrite[T, Float]()
    } else if (t == DataType.DT_DOUBLE) {
      new TensorArrayWrite[T, Float]()
    } else if (t == DataType.DT_INT32) {
      new TensorArrayWrite[T, Int]()
    } else {
      throw new UnsupportedOperationException(
        s"Not support load TensorArrayWriteV3 with data type $t")
    }
  }
}

class TensorArraySizeV3 extends TensorflowOpsLoader {
  override def build[T: ClassManifest](
    nodeDef: NodeDef,
    byteOrder: ByteOrder,
    context: Context[T]
  )(implicit ev: TensorNumericMath.TensorNumeric[T]): Module[T] = {
    new TensorArraySize[T]()
  }
}


class StackPopV2 extends TensorflowOpsLoader {

  import Utils._

  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder, context: Context[T])
    (implicit ev: TensorNumeric[T]): Module[T] = {
    Identity()
  }
}

class StackPushV2 extends TensorflowOpsLoader {

  import Utils._

  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder, context: Context[T])
    (implicit ev: TensorNumeric[T]): Module[T] = {
    Identity()
  }
}

class StackV2 extends TensorflowOpsLoader {

  import Utils._

  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder, context: Context[T])
    (implicit ev: TensorNumeric[T]): Module[T] = {
    Identity()
  }
}
