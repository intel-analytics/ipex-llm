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

import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.bigdl.transform.vision.image.{FeatureTransformer, ImageFeature}
import org.apache.log4j.Logger
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

import scala.util.Random

/**
 * Resize image
 * @param resizeH height after resize
 * @param resizeW width after resize
 * @param resizeMode if resizeMode = -1, random select a mode from
 * (Imgproc.INTER_LINEAR, Imgproc.INTER_CUBIC, Imgproc.INTER_AREA,
 *                   Imgproc.INTER_NEAREST, Imgproc.INTER_LANCZOS4)
 * @param useScaleFactor if true, scale factor fx and fy is used, fx = fy = 0
 * note that the result of the following are different
 * Imgproc.resize(mat, mat, new Size(resizeWH, resizeWH), 0, 0, Imgproc.INTER_LINEAR)
 * Imgproc.resize(mat, mat, new Size(resizeWH, resizeWH))
 */
class Resize(resizeH: Int, resizeW: Int,
  resizeMode: Int = Imgproc.INTER_LINEAR,
  useScaleFactor: Boolean = true)
  extends FeatureTransformer {

  private val interpMethods = Array(Imgproc.INTER_LINEAR, Imgproc.INTER_CUBIC, Imgproc.INTER_AREA,
    Imgproc.INTER_NEAREST, Imgproc.INTER_LANCZOS4)

  override def transformMat(feature: ImageFeature): Unit = {
    val interpMethod = if (resizeMode == -1) {
      interpMethods(new Random().nextInt(interpMethods.length))
    } else {
      resizeMode
    }
    Resize.transform(feature.opencvMat(), feature.opencvMat(), resizeW, resizeH, interpMethod,
      useScaleFactor)
  }
}

object Resize {
  val logger = Logger.getLogger(getClass)

  def apply(resizeH: Int, resizeW: Int,
    resizeMode: Int = Imgproc.INTER_LINEAR, useScaleFactor: Boolean = true): Resize =
    new Resize(resizeH, resizeW, resizeMode, useScaleFactor)

  def transform(input: OpenCVMat, output: OpenCVMat, resizeW: Int, resizeH: Int,
                mode: Int = Imgproc.INTER_LINEAR, useScaleFactor: Boolean = true)
  : OpenCVMat = {
    if (useScaleFactor) {
      Imgproc.resize(input, output, new Size(resizeW, resizeH), 0, 0, mode)
    } else {
      Imgproc.resize(input, output, new Size(resizeW, resizeH))
    }
    output
  }
}

/**
 * Resize the image, keep the aspect ratio. scale according to the short edge
 * @param minSize scale size, apply to short edge
 * @param scaleMultipleOf make the scaled size multiple of some value
 * @param maxSize max size after scale
 * @param resizeMode if resizeMode = -1, random select a mode from
 * (Imgproc.INTER_LINEAR, Imgproc.INTER_CUBIC, Imgproc.INTER_AREA,
 *                   Imgproc.INTER_NEAREST, Imgproc.INTER_LANCZOS4)
 * @param useScaleFactor if true, scale factor fx and fy is used, fx = fy = 0
 * @param minScale control the minimum scale up for image
 */
class AspectScale(minSize: Int,
  scaleMultipleOf: Int = 1,
  maxSize: Int = 1000,
  resizeMode: Int = Imgproc.INTER_LINEAR,
  useScaleFactor: Boolean = true,
  minScale: Option[Float] = None)
  extends FeatureTransformer {

  override def transformMat(feature: ImageFeature): Unit = {
    val (height, width) = AspectScale.getHeightWidthAfterRatioScale(feature.opencvMat(),
      minSize, maxSize, scaleMultipleOf, minScale)
    Resize.transform(feature.opencvMat(), feature.opencvMat(),
      width, height, resizeMode, useScaleFactor)
  }
}

object AspectScale {

  def apply(minSize: Int,
    scaleMultipleOf: Int = 1,
    maxSize: Int = 1000,
    mode: Int = Imgproc.INTER_LINEAR,
    useScaleFactor: Boolean = true,
    minScale: Option[Float] = None): AspectScale =
    new AspectScale(minSize, scaleMultipleOf, maxSize, mode, useScaleFactor, minScale)
  /**
   * get the width and height of scaled image
   * @param img original image
   */
  def getHeightWidthAfterRatioScale(img: OpenCVMat, scaleTo: Float,
    maxSize: Int, scaleMultipleOf: Int, minScale: Option[Float] = None): (Int, Int) = {
    val imSizeMin = Math.min(img.width(), img.height())
    val imSizeMax = Math.max(img.width(), img.height())
    var imScale = scaleTo.toFloat / imSizeMin.toFloat
    if (minScale.isDefined) {
      imScale = Math.max(minScale.get, imScale)
    }
    // Prevent the biggest axis from being more than MAX_SIZE
    if (Math.round(imScale * imSizeMax) > maxSize) {
      imScale = maxSize / imSizeMax.toFloat
    }

    var imScaleH, imScaleW = imScale
    if (scaleMultipleOf > 1) {
      imScaleH = (Math.floor(img.height() * imScale / scaleMultipleOf) *
        scaleMultipleOf / img.height()).toFloat
      imScaleW = (Math.floor(img.width() * imScale / scaleMultipleOf) *
        scaleMultipleOf / img.width()).toFloat
    }

    val width = imScaleW * img.width()
    val height = imScaleH * img.height()
    (height.round, width.round)
  }
}


/**
 * resize the image by randomly choosing a scale
 * @param scales array of scale options that for random choice
 * @param scaleMultipleOf Resize test images so that its width and height are multiples of
 * @param maxSize Max pixel size of the longest side of a scaled input image
 */
class RandomAspectScale(scales: Array[Int], scaleMultipleOf: Int = 1,
  maxSize: Int = 1000) extends FeatureTransformer {

  override def transformMat(feature: ImageFeature): Unit = {
    val scaleTo = scales(Random.nextInt(scales.length))
    val (height, width) = AspectScale.getHeightWidthAfterRatioScale(feature.opencvMat(),
      scaleTo, maxSize, scaleMultipleOf)
    Resize.transform(feature.opencvMat(), feature.opencvMat(), width, height)
  }
}

object RandomAspectScale {
  def apply(scales: Array[Int], scaleMultipleOf: Int = 1,
    maxSize: Int = 1000): RandomAspectScale =
    new RandomAspectScale(scales, scaleMultipleOf, maxSize)
}

