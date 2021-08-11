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

import breeze.numerics.sqrt
import org.opencv.core.{CvType, Mat, Rect}
import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.dataset.image.LabeledBGRImage
import com.intel.analytics.bigdl.opencv.OpenCV
import org.opencv.imgproc.Imgproc

import scala.collection.Iterator
import com.intel.analytics.bigdl.opencv
import com.intel.analytics.bigdl.transform.vision.image.{FeatureTransformer, ImageFeature}
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import org.apache.spark.ml
import org.apache.spark.ml.feature
import org.opencv.core.Size

object RandomAlterAspect {
  def apply(min_area_ratio: Float = 0.08f,
            max_area_ratio: Int = 1,
            min_aspect_ratio_change: Float = 0.75f,
            interp_mode: String = "CUBIC",
            cropLength: Int = 224): RandomAlterAspect = {
    OpenCV.isOpenCVLoaded
    new RandomAlterAspect(min_area_ratio, max_area_ratio,
      min_aspect_ratio_change, interp_mode, cropLength)
  }
}

/**
 * Apply random crop based on area ratio and resize to cropLenth size
 * @param min_area_ratio  min area ratio
 * @param max_area_ratio  max area ratio
 * @param min_aspect_ratio_change factor applied to ratio area
 * @param interp_mode   interp mode applied in resize
 * @param cropLength final size resized to
 */
class RandomAlterAspect(min_area_ratio: Float = 0.08f,
                           max_area_ratio: Int = 1,
                           min_aspect_ratio_change: Float = 0.75f,
                           interp_mode: String = "CUBIC",
                           cropLength: Int = 224)
  extends FeatureTransformer {

  import com.intel.analytics.bigdl.utils.RandomGenerator.RNG

  @inline
  private def randRatio(min: Float, max: Float): Float = {
    val res = (RNG.uniform(1e-2, (max - min) * 1000 + 1) + min * 1000) / 1000
    res.toFloat
  }

  override protected def transformMat(feature: ImageFeature): Unit = {
    val h = feature.opencvMat().size().height
    val w = feature.opencvMat().size().width
    val area = h * w

    require(min_area_ratio <= max_area_ratio, "min_area_ratio should <= max_area_ratio")

    var attempt = 0
    while (attempt < 10) {
      val area_ratio = randRatio(min_area_ratio, max_area_ratio)
      val aspect_ratio_change = randRatio(min_aspect_ratio_change, 1 / min_aspect_ratio_change)
      val new_area = area_ratio * area
      var new_h = (sqrt(new_area) * aspect_ratio_change).toInt
      var new_w = (sqrt(new_area) / aspect_ratio_change).toInt
      if (randRatio(0, 1) < 0.5) {
        val tmp = new_h
        new_h = new_w
        new_w = tmp
      }
      if (new_h <= h && new_w <= w) {
        val y = RNG.uniform(1e-2, h - new_h + 1).toInt
        val x = RNG.uniform(1e-2, w - new_w + 1).toInt
        Crop.transform(feature.opencvMat(),
          feature.opencvMat(), x, y, x + new_w, y + new_h, false, false)

        Imgproc.resize(feature.opencvMat(), feature.opencvMat(),
            new Size(cropLength, cropLength), 0, 0, 2)
        attempt = 100
      }
      attempt += 1
    }
    if (attempt < 20) {
      val (new_h, new_w) = resizeImagePerShorterSize(feature.opencvMat(), cropLength)
      Imgproc.resize(feature.opencvMat(),
        feature.opencvMat(), new Size(cropLength, cropLength), 0, 0, 2)
    }
  }

  private def resizeImagePerShorterSize(img: Mat, shorter_size: Int) : (Int, Int) = {
    val h = img.size().height
    val w = img.size().width
    var new_h = shorter_size
    var new_w = shorter_size

    if (h < w) {
      new_w = (w / h * shorter_size).toInt
    } else {
      new_h = (h / w * shorter_size).toInt
    }
    (new_h, new_w)
  }
}
