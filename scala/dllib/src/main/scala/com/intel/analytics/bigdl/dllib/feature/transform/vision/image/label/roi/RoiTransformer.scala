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

import com.intel.analytics.bigdl.transform.vision.image.util.{BboxUtil, BoundingBox}
import com.intel.analytics.bigdl.transform.vision.image.{FeatureTransformer, ImageFeature}

import scala.collection.mutable.ArrayBuffer

/**
 * Normalize Roi to [0, 1]
 */
case class RoiNormalize() extends FeatureTransformer {
  override def transformMat(feature: ImageFeature): Unit = {
    val height = feature.getHeight()
    val width = feature.getWidth()
    val label = feature(ImageFeature.label).asInstanceOf[RoiLabel]
    BboxUtil.scaleBBox(label.bboxes, 1.0f / height, 1.0f / width)
  }
}

/**
 * horizontally flip the roi
 * @param normalized whether the roi is normalized, i.e. in range [0, 1]
 */
case class RoiHFlip(normalized: Boolean = true) extends FeatureTransformer {
  override def transformMat(feature: ImageFeature): Unit = {
    val roiLabel = feature.getLabel[RoiLabel]
    var i = 1
    val width = if (normalized) 1 else feature.getWidth()
    while (i <= roiLabel.size()) {
      val x1 = width - roiLabel.bboxes.valueAt(i, 1)
      roiLabel.bboxes.setValue(i, 1, width - roiLabel.bboxes.valueAt(i, 3))
      roiLabel.bboxes.setValue(i, 3, x1)
      i += 1
    }
  }
}

/**
 * resize the roi according to scale
 * @param normalized whether the roi is normalized, i.e. in range [0, 1]
 */
case class RoiResize(normalized: Boolean = false) extends FeatureTransformer {
  override def transformMat(feature: ImageFeature): Unit = {
    if (!normalized) {
      val scaledW = feature.getWidth().toFloat / feature.getOriginalWidth
      val scaledH = feature.getHeight().toFloat / feature.getOriginalHeight
      val target = feature.getLabel[RoiLabel]
      BboxUtil.scaleBBox(target.bboxes, scaledH, scaledW)
    }
  }
}

/**
 * Project gt boxes onto the coordinate system defined by image boundary
 * @param needMeetCenterConstraint whether need to meet center constraint, i.e., the center of
 * gt box need be within image boundary
 */
case class RoiProject(needMeetCenterConstraint: Boolean = true) extends FeatureTransformer {
  val transformedAnnot = new ArrayBuffer[BoundingBox]()
  override def transformMat(feature: ImageFeature): Unit = {
    val imageBoundary = feature[BoundingBox](ImageFeature.boundingBox)
    if (!imageBoundary.normalized) {
      imageBoundary.scaleBox(1.0f / feature.getHeight(), 1f / feature.getWidth(), imageBoundary)
    }
    val target = feature[RoiLabel](ImageFeature.label)
    transformedAnnot.clear()
    // Transform the annotation according to bounding box.
    var i = 1
    while (i <= target.size()) {
      val gtBoxes = BoundingBox(target.bboxes.valueAt(i, 1),
        target.bboxes.valueAt(i, 2),
        target.bboxes.valueAt(i, 3),
        target.bboxes.valueAt(i, 4))
      if (!needMeetCenterConstraint ||
        imageBoundary.meetEmitCenterConstraint(gtBoxes)) {
        val transformedBox = new BoundingBox()
        if (imageBoundary.projectBbox(gtBoxes, transformedBox)) {
          transformedBox.setLabel(target.classes.valueAt(1, i))
          transformedBox.setDifficult(target.classes.valueAt(2, i))
          transformedAnnot.append(transformedBox)
        }
      }
      i += 1
    }
    // write the transformed annotation back to target
    target.bboxes.resize(transformedAnnot.length, 4)
    target.classes.resize(2, transformedAnnot.length)

    i = 1
    while (i <= transformedAnnot.length) {
      target.bboxes.setValue(i, 1, transformedAnnot(i - 1).x1)
      target.bboxes.setValue(i, 2, transformedAnnot(i - 1).y1)
      target.bboxes.setValue(i, 3, transformedAnnot(i - 1).x2)
      target.bboxes.setValue(i, 4, transformedAnnot(i - 1).y2)
      target.classes.setValue(1, i, transformedAnnot(i - 1).label)
      target.classes.setValue(2, i, transformedAnnot(i - 1).difficult)
      i += 1
    }
  }
}
