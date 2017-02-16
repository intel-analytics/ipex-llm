/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.dataset.roiimage

import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import javax.imageio.ImageIO

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.dataset.image.BGRImage
import com.intel.analytics.bigdl.tensor.Tensor

import scala.util.Random

/**
 * resize the image by randomly choosing a scale, resize the target if exists
 * @param scales          array of scale options that for random choice
 * @param scaleMultipleOf Resize test images so that its width and height are multiples of
 * @param resizeRois      whether to resize target bboxes
 * @param maxSize         Max pixel size of the longest side of a scaled input image
 */
class RoiImageResizer(scales: Array[Int], scaleMultipleOf: Int, resizeRois: Boolean,
  maxSize: Float = 1000f)
  extends Transformer[RoiByteImage, RoiImage] {

  val imageWithRoi = new RoiImage()

  override def apply(prev: Iterator[RoiByteImage]): Iterator[RoiImage] = {
    prev.map(roiByteImage => {
      // convert byte array back to BufferedImage
      val img = ImageIO.read(new ByteArrayInputStream(roiByteImage.data,
        0, roiByteImage.dataLength))
      getWidthHeightAfterRatioScale(img, imageWithRoi.imInfo)
      val scaledImage = BGRImage.resizeImage(img, imageWithRoi.imInfo.valueAt(2).toInt,
        imageWithRoi.imInfo.valueAt(1).toInt)
      imageWithRoi.copy(scaledImage)
      imageWithRoi.path = roiByteImage.path
      if (resizeRois) {
        require(roiByteImage.target.isDefined, "target is not defined")
        imageWithRoi.target = resizeRois(imageWithRoi.scaledW, imageWithRoi.scaledW,
          roiByteImage.target.get)
      } else {
        imageWithRoi.target = roiByteImage.target
      }
      imageWithRoi
    })
  }


  /**
   * get the width and height of scaled image
   * @param img original image
   * @return imageInfo (scaledHeight, scaledWidth, scaleRatioH, scaleRatioW)
   */
  def getWidthHeightAfterRatioScale(img: BufferedImage, imInfo: Tensor[Float]): Tensor[Float] = {
    val scaleTo = scales(Random.nextInt(scales.length))
    val imSizeMin = Math.min(img.getWidth, img.getHeight)
    val imSizeMax = Math.max(img.getWidth, img.getHeight)
    var im_scale = scaleTo.toFloat / imSizeMin.toFloat
    // Prevent the biggest axis from being more than MAX_SIZE
    if (Math.round(im_scale * imSizeMax) > maxSize) {
      im_scale = maxSize / imSizeMax.toFloat
    }

    val imScaleH = (Math.floor(img.getHeight * im_scale / scaleMultipleOf) *
      scaleMultipleOf / img.getHeight).toFloat
    val imScaleW = (Math.floor(img.getWidth * im_scale / scaleMultipleOf) *
      scaleMultipleOf / img.getWidth).toFloat
    imInfo.setValue(1, imScaleH * img.getHeight)
    imInfo.setValue(2, imScaleW * img.getWidth)
    imInfo.setValue(3, imScaleH)
    imInfo.setValue(4, imScaleW)
  }

  /**
   * resize the ground truth rois
   * @param scaledH
   * @param scaledW
   * @param target
   * @return
   */
  def resizeRois(scaledH: Float, scaledW: Float, target: Target): Option[Target] = {
    var resizedTarget: Option[Target] = None
    if (resizedTarget != null) {
      val gtInds = target.classes.storage().array().zip(Stream from 1)
        .filter(x => x._1 != 0).map(x => x._2)
      val resizedBoxes = Tensor[Float](gtInds.length, 5)
      var i = 0
      while (i < gtInds.length) {
        resizedBoxes.setValue(i + 1, 1, target.bboxes.valueAt(gtInds(i), 1) * scaledH)
        resizedBoxes.setValue(i + 1, 2, target.bboxes.valueAt(gtInds(i), 2) * scaledW)
        resizedBoxes.setValue(i + 1, 3, target.bboxes.valueAt(gtInds(i), 3) * scaledH)
        resizedBoxes.setValue(i + 1, 4, target.bboxes.valueAt(gtInds(i), 4) * scaledW)
        resizedBoxes.setValue(i + 1, 5, target.classes.valueAt(gtInds(i)))
        i += 1
      }
      resizedTarget = Some(Target(null, resizedBoxes))
    }
    resizedTarget
  }
}

object RoiImageResizer {
  def apply(scales: Array[Int], scaleMultipleOf: Int, resizeRois: Boolean): RoiImageResizer =
    new RoiImageResizer(scales, scaleMultipleOf, resizeRois)
}
