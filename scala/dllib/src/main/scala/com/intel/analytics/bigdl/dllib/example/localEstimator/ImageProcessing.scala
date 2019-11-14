/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.examples.localEstimator

import java.awt.image.{BufferedImage, DataBufferByte}
import java.nio.ByteBuffer

import com.intel.analytics.bigdl.dataset.image.{BGRImage, GreyImage, LabeledBGRImage, LabeledGreyImage}
import com.intel.analytics.bigdl.dataset.{ByteRecord, MiniBatch}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.RandomGenerator
import com.intel.analytics.zoo.pipeline.estimator.EstimateSupportive

trait ImageProcessing extends EstimateSupportive {
  val normalize = 255f

  def bytesToGreyImage(record: ByteRecord, row: Int, col: Int): LabeledGreyImage = {
    val buffer = new LabeledGreyImage(row, col)
    require(row * col == record.data.length)
    buffer.setLabel(record.label).copy(record.data, 255.0f)
  }

  def greyImgNormalize(labeledGreyImage: LabeledGreyImage,
                       mean: Double,
                       std: Double): LabeledGreyImage = {
    val content = labeledGreyImage.content
    var i = 0
    while (i < content.length) {
      content(i) = ((content(i) - mean) / std).toFloat
      i += 1
    }
    labeledGreyImage
  }

  def bytesToLabeledBGRImage(record : ByteRecord, resizeW : Int, resizeH : Int): LabeledBGRImage = {
    val buffer = new LabeledBGRImage()
    val imgData = if (resizeW == -1) {
      record.data
    } else {
      val rawData = record.data
      val imgBuffer = ByteBuffer.wrap(rawData)
      val width = imgBuffer.getInt
      val height = imgBuffer.getInt
      val bufferedImage: BufferedImage =
        new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR)
      val outputImagePixelData =
        bufferedImage.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData
      System.arraycopy(imgBuffer.array(), 8, outputImagePixelData, 0, outputImagePixelData.length)
      BGRImage.resizeImage(bufferedImage, resizeW, resizeH)
    }
    buffer.copy(imgData, normalize).setLabel(record.label)
  }

  def bgrImgNormalize(labeledBGRImage: LabeledBGRImage,
                      mean: (Double, Double, Double),
                      std: (Double, Double, Double)): LabeledBGRImage = {
    val (meanR, meanG, meanB) = mean
    val (stdR, stdG, stdB) = std
    val content = labeledBGRImage.content
    require(content.length % 3 == 0)
    var i = 0
    while (i < content.length) {
      content(i + 2) = ((content(i + 2) - meanR) / stdR).toFloat
      content(i + 1) = ((content(i + 1) - meanG) / stdG).toFloat
      content(i + 0) = ((content(i + 0) - meanB) / stdB).toFloat
      i += 3
    }
    labeledBGRImage
  }

  def hFlip(labeledBGRImage: LabeledBGRImage, threshold: Double): LabeledBGRImage = {
    if (RandomGenerator.RNG.uniform(0, 1) >= threshold) {
      labeledBGRImage.hflip()
    } else {
      labeledBGRImage
    }
  }

  def bgrImageRandomCrop(labeledBGRImage: LabeledBGRImage,
                         cropWidth: Int,
                         cropHeight: Int,
                         padding: Int): LabeledBGRImage = {
    val curImg = padding > 0 match {
      case true =>
        val widthTmp = labeledBGRImage.width()
        val heightTmp = labeledBGRImage.height()
        val sourceTmp = labeledBGRImage.content
        val padWidth = widthTmp + 2 * padding
        val padHeight = heightTmp + 2 * padding
        val temp = new LabeledBGRImage(padWidth, padHeight)
        val tempBuffer = temp.content
        val startIndex = (padding + padding * padWidth) * 3
        val frameLength = widthTmp * heightTmp
        var i = 0
        while (i < frameLength) {
          tempBuffer(startIndex +
            ((i / widthTmp) * padWidth + (i % widthTmp)) * 3 + 2) = sourceTmp(i * 3 + 2)
          tempBuffer(startIndex +
            ((i / widthTmp) * padWidth + (i % widthTmp)) * 3 + 1) = sourceTmp(i * 3 + 1)
          tempBuffer(startIndex +
            ((i / widthTmp) * padWidth + (i % widthTmp)) * 3) = sourceTmp(i * 3)
          i += 1
        }
        temp.setLabel(labeledBGRImage.label())
        temp
      case _ => labeledBGRImage
    }

    val buffer = new LabeledBGRImage(cropWidth, cropHeight)
    val width = curImg.width()
    val height = curImg.height()
    val source = curImg.content
    val startW = RandomGenerator.RNG.uniform(0, width - cropWidth).toInt
    val startH = RandomGenerator.RNG.uniform(0, height - cropHeight).toInt
    val startIndex = (startW + startH * width) * 3
    val frameLength = cropWidth * cropHeight
    val target = buffer.content
    var i = 0
    while (i < frameLength) {
      target(i * 3 + 2) =
        source(startIndex + ((i / cropWidth) * width + (i % cropWidth)) * 3 + 2)
      target(i * 3 + 1) =
        source(startIndex + ((i / cropWidth) * width + (i % cropWidth)) * 3 + 1)
      target(i * 3) =
        source(startIndex + ((i / cropWidth) * width + (i % cropWidth)) * 3)
      i += 1
    }
    buffer.setLabel(curImg.label())
    buffer
  }

  def labeledGreyImageToMiniBatch(images: Array[LabeledGreyImage]): MiniBatch[Float] = {
    val height = images(0).height()
    val width = images(0).width()
    val n = images.length

    val featureData = new Array[Float](n * height * width)
    val labelData = new Array[Float](n)
    val featureTensor: Tensor[Float] = Tensor[Float]()
    val labelTensor: Tensor[Float] = Tensor[Float]()

    var i = 0
    while(i < images.length) {
      val img = images(i)
      copyImage(img, featureData, i * width * height)
      labelData(i) = img.label()
      i += 1
      if (labelTensor.nElement() != i) {
        featureTensor.set(Storage[Float](featureData), 1, Array(i, height, width))
        labelTensor.set(Storage[Float](labelData), storageOffset = 1, sizes = Array(i))
      }
    }
    MiniBatch(featureTensor, labelTensor)
  }

  def labeledBGRImageToMiniBatch(images: Array[LabeledBGRImage]): MiniBatch[Float] = {
    val height = images(0).height()
    val width = images(0).width()
    val toRGB = true
    val n = images.length

    val featureData = new Array[Float](n * 3 * height * width)
    val labelData = new Array[Float](n)
    val featureTensor: Tensor[Float] = Tensor[Float]()
    val labelTensor: Tensor[Float] = Tensor[Float]()

    var i = 0
    while(i < images.length) {
      val img = images(i)
      img.copyTo(featureData, i * width * height * 3, toRGB)
      labelData(i) = img.label()
      i += 1
      if (labelTensor.nElement() != i) {
        featureTensor.set(Storage[Float](featureData),
          storageOffset = 1, sizes = Array(i, 3, height, width))
        labelTensor.set(Storage[Float](labelData),
          storageOffset = 1, sizes = Array(i))
      }
    }
    MiniBatch(featureTensor, labelTensor)
  }

  private def copyImage(img: GreyImage, storage: Array[Float], offset: Int): Unit = {
    val content = img.content
    val frameLength = img.width() * img.height()
    var j = 0
    while (j < frameLength) {
      storage(offset + j) = content(j)
      j += 1
    }
  }
}

object ImageProcessing extends ImageProcessing {
  val labeledGreyImageToMiniBatchTransformer =
    (data: Array[LabeledGreyImage]) => labeledGreyImageToMiniBatch(data)
  val labeledBGRImageToMiniBatchTransformer =
    (data: Array[LabeledBGRImage]) => labeledBGRImageToMiniBatch(data)
}
