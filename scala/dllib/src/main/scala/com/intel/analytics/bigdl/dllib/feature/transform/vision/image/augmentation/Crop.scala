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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.{FeatureTransformer, ImageFeature}
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.bigdl.transform.vision.image.util.{BboxUtil, BoundingBox}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.opencv.core.Rect

/**
 * Abstract crop transformer, other crop transformer need to override generateRoi
 *
 * @param normalized whether the roi is normalized
 * @param isClip whether to clip the roi to image boundaries
 */
abstract class Crop(normalized: Boolean = true, isClip: Boolean = true) extends FeatureTransformer {

  /**
   * how to generate crop roi
   * @param feature image feature
   * @return crop roi
   */
  def generateRoi(feature: ImageFeature): BoundingBox

  override def transformMat(feature: ImageFeature): Unit = {
    val cropBox = generateRoi(feature)
    Crop.transform(feature.opencvMat(), feature.opencvMat(),
      cropBox.x1, cropBox.y1, cropBox.x2, cropBox.y2, normalized, isClip)
    if (feature.hasLabel()) {
      feature(ImageFeature.boundingBox) = cropBox
    }
  }
}

object Crop {

  def transform(input: OpenCVMat, output: OpenCVMat,
    wStart: Float, hStart: Float, wEnd: Float, hEnd: Float, normalized: Boolean = true,
    isClip: Boolean = true): Unit = {
    val width = input.width
    val height = input.height
    var (x1, y1, x2, y2) = if (normalized) {
      // scale back to original size
      (wStart * width, hStart * height, wEnd * width, hEnd * height)
    } else {
      (wStart, hStart, wEnd, hEnd)
    }
    if (isClip) {
      // clip to image boundary
      x1 = Math.max(Math.min(x1, width), 0f)
      y1 = Math.max(Math.min(y1, height), 0f)
      x2 = Math.max(Math.min(x2, width), 0f)
      y2 = Math.max(Math.min(y2, height), 0f)
    }
    val rect = new Rect(x1.toInt, y1.toInt, (x2 - x1).toInt, (y2 - y1).toInt)
    val submat = input.submat(rect)
    submat.copyTo(output)
    submat.release()
  }
}

/**
 * Crop a `cropWidth` x `cropHeight` patch from center of image.
 * The patch size should be less than the image size.
 *
 * @param cropWidth width after crop
 * @param cropHeight height after crop
 * @param isClip whether to clip the roi to image boundaries
 */
class CenterCrop(cropWidth: Int, cropHeight: Int, isClip: Boolean) extends Crop(false, isClip) {
  override def generateRoi(feature: ImageFeature): BoundingBox = {
    val mat = feature.opencvMat()
    val height = mat.height().toFloat
    val width = mat.width().toFloat
    val startH = (height - cropHeight) / 2
    val startW = (width - cropWidth) / 2
    BoundingBox(startW, startH, startW + cropWidth, startH + cropHeight, false)
  }
}

object CenterCrop {
  def apply(cropWidth: Int, cropHeight: Int, isClip: Boolean = true)
  : CenterCrop = new CenterCrop(cropWidth, cropHeight, isClip)
}

/**
 * Random crop a `cropWidth` x `cropHeight` patch from an image.
 * The patch size should be less than the image size.
 *
 * @param cropWidth width after crop
 * @param cropHeight height after crop
 * @param isClip whether to clip the roi to image boundaries
 */
class RandomCrop(cropWidth: Int, cropHeight: Int, isClip: Boolean) extends Crop(false, isClip) {

  override def generateRoi(feature: ImageFeature): BoundingBox = {
    val mat = feature.opencvMat()
    val height = mat.height().toFloat
    val width = mat.width().toFloat
    val startH = math.ceil(RNG.uniform(1e-2, height - cropHeight)).toFloat
    val startW = math.ceil(RNG.uniform(1e-2, width - cropWidth)).toFloat
    BoundingBox(startW, startH, startW + cropWidth, startH + cropHeight, false)
  }
}

object RandomCrop {
  def apply(cropWidth: Int, cropHeight: Int, isClip: Boolean = true): RandomCrop =
    new RandomCrop(cropWidth, cropHeight, isClip)
}

/**
 * Crop a fixed area of image
 *
 * @param x1 start in width
 * @param y1 start in height
 * @param x2 end in width
 * @param y2 end in height
 * @param normalized whether args are normalized, i.e. in range [0, 1]
 * @param isClip whether to clip the roi to image boundaries
 */
class FixedCrop(x1: Float, y1: Float, x2: Float, y2: Float, normalized: Boolean,
  isClip: Boolean)
  extends Crop(normalized, isClip) {

  @transient private var cropBox: BoundingBox = _

  override def generateRoi(feature: ImageFeature): BoundingBox = {
    if (null == cropBox) cropBox = BoundingBox(x1, y1, x2, y2, normalized)
    cropBox
  }
}

object FixedCrop {
  def apply(x1: Float, y1: Float, x2: Float, y2: Float, normalized: Boolean,
    isClip: Boolean = true)
  : FixedCrop = new FixedCrop(x1, y1, x2, y2, normalized, isClip)
}

/**
 * Crop from object detections, each image should has a tensor detection,
 * which is stored in ImageFeature
 *
 * @param roiKey roiKey that map a tensor detection
 * @param normalized whether is detection is normalized, i.e. in range [0, 1]
 */
class DetectionCrop(roiKey: String, normalized: Boolean = true) extends Crop(normalized, true) {

  override def generateRoi(feature: ImageFeature): BoundingBox = {
    require(feature(roiKey).isInstanceOf[Tensor[Float]], "currently only support tensor detection")
    var roi = feature(roiKey).asInstanceOf[Tensor[Float]]
    if (roi.dim() == 1) {
      roi = BboxUtil.decodeRois(roi)
    }
    if (roi.nElement() >= 6 && roi.dim() == 2) {
      BoundingBox(roi.valueAt(1, 3), roi.valueAt(1, 4),
        roi.valueAt(1, 5), roi.valueAt(1, 6), normalized)
    } else {
      BoundingBox(0, 0, 1, 1, normalized)
    }
  }
}

object DetectionCrop {
  def apply(roiKey: String, normalized: Boolean = true): DetectionCrop =
    new DetectionCrop(roiKey, normalized)
}


