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

package com.intel.analytics.bigdl.transform.vision.image.label.roi

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{T, Table}

/**
 * image target with classes and bounding boxes
 *
 * @param classes N (class labels) or 2 * N, the first row is class labels,
 * the second line is difficults
 * @param bboxes N * 4
 */
case class RoiLabel(classes: Tensor[Float], bboxes: Tensor[Float]) {
  def copy(target: RoiLabel): Unit = {
    classes.resizeAs(target.classes).copy(target.classes)
    bboxes.resizeAs(target.bboxes).copy(target.bboxes)
  }

  if (classes.dim() == 1) {
    require(classes.size(1) == bboxes.size(1), "the number of classes should be" +
      " equal to the number of bounding box numbers")
  } else if (classes.nElement() > 0 && classes.dim() == 2) {
    require(classes.size(2) == bboxes.size(1), s"the number of classes ${ classes.size(2) }" +
      s"should be equal to the number of bounding box numbers ${ bboxes.size(1) }")
  }


  def toTable: Table = {
    val table = T()
    table.insert(classes)
    table.insert(bboxes)
  }

  def size(): Int = {
    if (bboxes.nElement() < 4) 0 else bboxes.size(1)
  }
}

object RoiLabel {
  def fromTensor(tensor: Tensor[Float]): RoiLabel = {
    val label = tensor.narrow(2, 1, 2).transpose(1, 2).contiguous()
    val rois = tensor.narrow(2, 3, 4)
    RoiLabel(label, rois)
  }
}

