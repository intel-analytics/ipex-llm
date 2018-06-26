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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.tf.Tensorflow._
import com.intel.analytics.bigdl.utils.tf.TensorflowSpecHelper
import org.tensorflow.framework.{DataType, NodeDef}

class StridedSliceSpec extends TensorflowSpecHelper {

  "StridedSlice forward float" should "be correct" in {
    compare[Float](
      NodeDef.newBuilder()
        .setName("StridedSliceTest")
        .setOp(s"StridedSlice")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("Index", typeAttr(DataType.DT_INT32))
        .putAttr("begin_mask", intAttr(0))
        .putAttr("end_mask", intAttr(0))
        .putAttr("ellipsis_mask", intAttr(0))
        .putAttr("new_axis_mask", intAttr(0))
        .putAttr("shrink_axis_mask", intAttr(1)),
      Seq(Tensor[Float](T(40, 128, 64)), Tensor[Int](T(0)),
        Tensor[Int](T(1)), Tensor[Int](T(1))),
      0
    )

    compare[Float](
      NodeDef.newBuilder()
        .setName("StridedSliceTest")
        .setOp(s"StridedSlice")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("Index", typeAttr(DataType.DT_INT32))
        .putAttr("begin_mask", intAttr(0))
        .putAttr("end_mask", intAttr(0))
        .putAttr("ellipsis_mask", intAttr(0))
        .putAttr("new_axis_mask", intAttr(0))
        .putAttr("shrink_axis_mask", intAttr(1)),
      Seq(Tensor[Float](T(40, 128, 64)), Tensor[Int](T(1)),
        Tensor[Int](T(2)), Tensor[Int](T(1))),
      0
    )

    compare[Float](
      NodeDef.newBuilder()
        .setName("StridedSliceTest")
        .setOp(s"StridedSlice")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("Index", typeAttr(DataType.DT_INT32))
        .putAttr("begin_mask", intAttr(0))
        .putAttr("end_mask", intAttr(0))
        .putAttr("ellipsis_mask", intAttr(0))
        .putAttr("new_axis_mask", intAttr(0))
        .putAttr("shrink_axis_mask", intAttr(0)),
      Seq(Tensor[Float](T(
        T(T(1, 1, 1), T(2, 2, 2)),
        T(T(3, 3, 3), T(4, 4, 4)),
        T(T(5, 5, 5), T(6, 6, 6))
      )), Tensor[Int](T(1, 0, 0)),
        Tensor[Int](T(2, 1, 3)), Tensor[Int](T(1, 1, 1))),
      0
    )

    compare[Float](
      NodeDef.newBuilder()
        .setName("StridedSliceTest")
        .setOp(s"StridedSlice")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("Index", typeAttr(DataType.DT_INT32))
        .putAttr("begin_mask", intAttr(5))
        .putAttr("end_mask", intAttr(5))
        .putAttr("ellipsis_mask", intAttr(0))
        .putAttr("new_axis_mask", intAttr(0))
        .putAttr("shrink_axis_mask", intAttr(2)),
      Seq(Tensor[Float](T(
        T(T(1, 1, 1), T(2, 2, 2)),
        T(T(3, 3, 3), T(4, 4, 4)),
        T(T(5, 5, 5), T(6, 6, 6)))
      ), Tensor[Int](T(0, -1, 0)),
        Tensor[Int](T(0, 0, 0)), Tensor[Int](T(1, 1, 1))),
      0
    )

    compare[Float](
      NodeDef.newBuilder()
        .setName("StridedSliceTest")
        .setOp(s"StridedSlice")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("Index", typeAttr(DataType.DT_INT32))
        .putAttr("begin_mask", intAttr(5))
        .putAttr("end_mask", intAttr(5))
        .putAttr("ellipsis_mask", intAttr(0))
        .putAttr("new_axis_mask", intAttr(0))
        .putAttr("shrink_axis_mask", intAttr(2)),
      Seq(Tensor[Float](T(
        T(T(1, 1, 1), T(2, 2, 2)),
        T(T(3, 3, 3), T(4, 4, 4)),
        T(T(5, 5, 5), T(6, 6, 6)))
      ), Tensor[Int](T(0, 1, 0)),
        Tensor[Int](T(0, 0, 0)), Tensor[Int](T(1, 1, 1))),
      0
    )
  }

  "StridedSlice forward int" should "be correct" in {
    compare[Int](
      NodeDef.newBuilder()
        .setName("StridedSliceTest")
        .setOp(s"StridedSlice")
        .putAttr("T", typeAttr(DataType.DT_INT32))
        .putAttr("Index", typeAttr(DataType.DT_INT32))
        .putAttr("begin_mask", intAttr(5))
        .putAttr("end_mask", intAttr(5))
        .putAttr("ellipsis_mask", intAttr(0))
        .putAttr("new_axis_mask", intAttr(0))
        .putAttr("shrink_axis_mask", intAttr(2)),
      Seq(Tensor[Int](T(
        T(T(1, 1, 1), T(2, 2, 2)),
        T(T(3, 3, 3), T(4, 4, 4)),
        T(T(5, 5, 5), T(6, 6, 6)))
      ), Tensor[Int](T(0, -1, 0)),
        Tensor[Int](T(0, 0, 0)), Tensor[Int](T(1, 1, 1))),
      0
    )

    compare[Int](
      NodeDef.newBuilder()
        .setName("StridedSliceTest")
        .setOp(s"StridedSlice")
        .putAttr("T", typeAttr(DataType.DT_INT32))
        .putAttr("Index", typeAttr(DataType.DT_INT32))
        .putAttr("begin_mask", intAttr(1))
        .putAttr("end_mask", intAttr(1))
        .putAttr("ellipsis_mask", intAttr(0))
        .putAttr("new_axis_mask", intAttr(0))
        .putAttr("shrink_axis_mask", intAttr(2)),
      Seq(Tensor[Int](T(
        T(T(1, 1, 1), T(2, 2, 2)),
        T(T(3, 3, 3), T(4, 4, 4)),
        T(T(5, 5, 5), T(6, 6, 6)))
      ), Tensor[Int](T(0, -1, 0)),
        Tensor[Int](T(0, 0, 2)), Tensor[Int](T(1, 1, 1))),
      0
    )

    compare[Int](
      NodeDef.newBuilder()
        .setName("StridedSliceTest")
        .setOp(s"StridedSlice")
        .putAttr("T", typeAttr(DataType.DT_INT32))
        .putAttr("Index", typeAttr(DataType.DT_INT32))
        .putAttr("begin_mask", intAttr(2))
        .putAttr("end_mask", intAttr(2))
        .putAttr("ellipsis_mask", intAttr(0))
        .putAttr("new_axis_mask", intAttr(0))
        .putAttr("shrink_axis_mask", intAttr(4)),
      Seq(Tensor[Int](T(
        T(T(1, 1, 1), T(2, 2, 2)),
        T(T(3, 3, 3), T(4, 4, 4)),
        T(T(5, 5, 5), T(6, 6, 6)))
      ), Tensor[Int](T(0, 0, -1)),
        Tensor[Int](T(1, 0, 0)), Tensor[Int](T(1, 1, 1))),
      0
    )
  }
}

