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

package com.intel.analytics.bigdl.transform.vision.image.augmentation

import com.intel.analytics.bigdl.dataset.segmentation.PolyMasks
import com.intel.analytics.bigdl.transform.vision.image.{FeatureTransformer, ImageFeature}
import com.intel.analytics.bigdl.transform.vision.image.label.roi.RoiLabel
import com.intel.analytics.bigdl.transform.vision.image.util.BboxUtil


object ScaleResize {
  /**
   * Scaling length and width of image feature to ensure that:
   * if maxSize is not set, the smaller one between width and length will be scaled to minSize.
   * if maxSize is set, the larger one will be scaled to maxSize or maxSize -1.
   * e.g. image feature height = 375, width = 500
   * case 1: minSize=100, maxSize=120, then new size (90, 120)
   * case 2: minSize=100, maxSize=-1, then new size (100, 133)
   * @param minSize the minimal size after resize
   * @param maxSize the maximal size after resize
   * @param resizeROI whether to resize roi, default false
   */
  def apply(minSize: Int, maxSize: Int = -1, resizeROI: Boolean = false): ScaleResize =
    new ScaleResize(minSize, maxSize, resizeROI)
}

class ScaleResize(minSize: Int, maxSize: Int = -1, resizeROI: Boolean = false)
  extends FeatureTransformer {
  private def getSize(sizeH: Int, sizeW: Int): (Int, Int) = {
    var size = minSize
    if (maxSize > 0) {
      val (minOrigSize, maxOrigSize) = if (sizeW > sizeH) (sizeH, sizeW) else (sizeW, sizeH)
      val thread = maxOrigSize.toFloat / minOrigSize * size
      if (thread > maxSize) size = math.round(maxSize * minOrigSize / maxOrigSize)
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
    Resize.transform(feature.opencvMat(), feature.opencvMat(), resizeW, resizeH,
      useScaleFactor = false)

    // resize roi label
    if (feature.hasLabel() && feature(ImageFeature.label).isInstanceOf[RoiLabel] && resizeROI) {
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

    val masks = feature.getLabel[RoiLabel].masks
    if (masks == null) return

    for (i <- 0 until masks.length) {
      val oneMask = masks(i)
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
        masks(i) = PolyMasks(poly, feature.getHeight(), feature.getWidth())
      }
    }
  }
}
