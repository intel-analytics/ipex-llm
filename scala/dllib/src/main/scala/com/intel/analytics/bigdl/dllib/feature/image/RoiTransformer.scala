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

package com.intel.analytics.bigdl.dllib.feature.image

import com.intel.analytics.bigdl.dllib.feature.transform.vision.image.label.roi.{RoiHFlip, RoiNormalize, RoiProject, RoiResize}
import com.intel.analytics.bigdl.dllib.feature.transform.vision.image.ImageFeature

/**
 * Normalize Roi to [0, 1]
 */
case class ImageRoiNormalize() extends ImageProcessing {
  private val internalRoiNormalize = RoiNormalize()
  override def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    internalRoiNormalize.apply(prev)
  }

  override def transformMat(feature: ImageFeature): Unit = {
    internalRoiNormalize.transformMat(feature)
  }
}

/**
 * horizontally flip the roi
 * @param normalized whether the roi is normalized, i.e. in range [0, 1]
 */
case class ImageRoiHFlip(normalized: Boolean = true) extends ImageProcessing {
  private val internalRoiHFlip = RoiHFlip(normalized)
  override def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    internalRoiHFlip.apply(prev)
  }

  override def transformMat(feature: ImageFeature): Unit = {
    internalRoiHFlip.transformMat(feature)
  }
}

/**
     * resize the roi according to scale
     * @param normalized whether the roi is normalized, i.e. in range [0, 1]
 */
case class ImageRoiResize(normalized: Boolean = false) extends ImageProcessing {
  private val internalRoiResize = RoiResize(normalized)
  override def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    internalRoiResize.apply(prev)
  }

  override def transformMat(feature: ImageFeature): Unit = {
    internalRoiResize.transformMat(feature)
  }
}

/**
 * Project gt boxes onto the coordinate system defined by image boundary
 * @param needMeetCenterConstraint whether need to meet center constraint, i.e., the center of
 * gt box need be within image boundary
 */
case class ImageRoiProject(needMeetCenterConstraint: Boolean = true) extends ImageProcessing {
  private val internalRoiProject = RoiProject(needMeetCenterConstraint)
  override def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    internalRoiProject.apply(prev)
  }

  override def transformMat(feature: ImageFeature): Unit = {
    internalRoiProject.transformMat(feature)
  }
}
