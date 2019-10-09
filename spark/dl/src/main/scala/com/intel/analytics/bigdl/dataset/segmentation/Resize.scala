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

package com.intel.analytics.bigdl.dataset.segmentation

import com.intel.analytics.bigdl.transform.vision.image.{FeatureTransformer, ImageFeature}
import com.intel.analytics.bigdl.transform.vision.image.label.roi.RoiLabel
import com.intel.analytics.bigdl.transform.vision.image.util.BboxUtil
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

class Resize(minSize: Int, maxSize: Int = -1, resizeROI: Boolean = false)
  extends FeatureTransformer {
  private def getSize(sizeH: Int, sizeW: Int): (Int, Int) = {
    var size = minSize
    if (maxSize > 0) {
      val min_original_size = math.min(sizeW, sizeH)
      val max_original_size = math.max(sizeW, sizeH)
      if (max_original_size / min_original_size * size > maxSize) {
        size = math.round(maxSize * min_original_size / max_original_size)
      }
    }

    if ((sizeW <= sizeH && sizeW == size) || (sizeH <= sizeW && sizeH == size)) {
      (sizeH, sizeW)
    } else if (sizeW < sizeH) {
      (size * sizeH / sizeW, size)
    } else {
      (size, size * sizeW / sizeH)
    }
  }

  override def transformMat(feature: ImageFeature): Unit = {
    val sizes = this.getSize(feature.getHeight(), feature.getWidth())
    val resizeH = sizes._1
    val resizeW = sizes._2
    Imgproc.resize(feature.opencvMat(), feature.opencvMat(), new Size(resizeW, resizeH))

    // resize roi label
    if (feature.hasLabel() && resizeROI) {
      // bbox resize
      resizeBbox(feature)
      // mask resize
      resizeMask(feature)
    }
  }

  private def resizeBbox(feature: ImageFeature): Unit = {
    val scaledW = feature.getWidth().toFloat / feature.getOriginalWidth
    val scaledH = feature.getHeight().toFloat / feature.getOriginalHeight
    val target = feature.getLabel[RoiLabel]
    BboxUtil.scaleBBox(target.bboxes, scaledH, scaledW)
  }

  private def resizeMask(feature: ImageFeature): Unit = {
    val scaledW = feature.getWidth().toFloat / feature.getOriginalWidth
    val scaledH = feature.getHeight().toFloat / feature.getOriginalHeight

    val mask = feature.getLabel[RoiLabel].masks
    for (i <- 0 to (mask.length - 1)) {
      val oneMask = mask(i)
      require(oneMask.isInstanceOf[PolyMasks],
        s"Only support poly mask resize, but get ${oneMask}")
      if (oneMask.isInstanceOf[PolyMasks]) {
        val polyMask = oneMask.asInstanceOf[PolyMasks]
        val poly = polyMask.poly
        for (i <- 0 to (poly.length - 1)) {
          val p = poly(i)
          for (j <- 0 to (p.length - 1)) {
            if (j % 2 == 0) p(j) *= scaledW // for x
            else p(j) *= scaledH // for y
          }
        }
        // change to resized mask
        mask(i) = PolyMasks(poly, feature.getHeight(), feature.getWidth())
      }
    }
  }
}

object Resize {
  def apply(minSize: Int, maxSize: Int = -1, resizeROI: Boolean = false)
    : Resize = new Resize(minSize, maxSize, resizeROI)
}
