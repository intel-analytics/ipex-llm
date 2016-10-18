/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
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
package com.intel.analytics.sparkdl.dataset

import com.intel.analytics.sparkdl.tensor.{Storage, Tensor}

trait NormalizerHelper {
  def checkSum(sum : Double) : Boolean = {
    sum < Double.MaxValue / (2 << 10) && sum > Double.MinValue / (2 << 10)
  }
}

class GreyImageNormalizer(dataSource: DataSource[GreyImage], samples: Int = -1)
  extends Transformer[GreyImage, GreyImage] with NormalizerHelper {

  private var mean: Double = 0
  private var std: Double = 0

  def getMean(): Double = mean

  def getStd(): Double = std

  init()

  private def init() = {
    var sum: Double = 0
    var total: Int = 0
    dataSource.shuffle()
    dataSource.reset()
    var i = 0
    while ((i < samples || samples < 0) && !dataSource.finished()) {
      val img = dataSource.next()
      img.content.foreach(e => {
        sum += e
        total += 1
      })
      i += 1
    }

    checkSum(sum)
    mean = sum / total

    sum = 0
    i = 0
    dataSource.reset()
    while ((i < samples || samples < 0) && !dataSource.finished()) {
      val img = dataSource.next()
      img.content.foreach(e => {
        val diff = e - mean
        sum += diff * diff
      })
      i += 1
    }
    checkSum(sum)
    std = math.sqrt(sum / total).toFloat
  }

  override def transform(prev: Iterator[GreyImage]): Iterator[GreyImage] = {
    prev.map(img => {
      var i = 0
      val content = img.content
      while (i < content.length) {
        content(i) = ((content(i) - mean) / std).toFloat
        i += 1
      }
      img
    })
  }
}

class RGBImageNormalizer(dataSource: DataSource[RGBImage], samples: Int = -1)
  extends Transformer[RGBImage, RGBImage] with NormalizerHelper {

  private var meanR: Double = 0
  private var stdR: Double = 0
  private var meanG: Double = 0
  private var stdG: Double = 0
  private var meanB: Double = 0
  private var stdB: Double = 0

  def getMean(): (Double, Double, Double) = (meanB, meanG, meanR)

  def getStd(): (Double, Double, Double) = (stdB, stdG, stdR)

  init()

  private def init() = {
    var sumR: Double = 0
    var sumG: Double = 0
    var sumB: Double = 0
    var total: Int = 0
    dataSource.shuffle()
    dataSource.reset()
    var i = 0
    while ((i < samples || samples < 0) && !dataSource.finished()) {
      val content = dataSource.next().content
      require(content.length % 3 == 0)
      var j = 0
      while (j < content.length) {
        sumR += content(j + 2)
        sumG += content(j + 1)
        sumB += content(j + 0)
        total += 1
        j += 3
      }
      i += 1
    }

    require(checkSum(sumR) && checkSum(sumG) & checkSum(sumB))
    meanR = (sumR / total).toFloat
    meanG = (sumG / total).toFloat
    meanB = (sumB / total).toFloat
    sumR = 0
    sumG = 0
    sumB = 0
    i = 0
    dataSource.reset()
    while ((i < samples || samples < 0) && !dataSource.finished()) {
      val content = dataSource.next().content
      var j = 0
      while (j < content.length) {
        val diffR = content(j + 2) - meanR
        val diffG = content(j + 1) - meanG
        val diffB = content(j + 0) - meanB
        sumR += diffR * diffR
        sumG += diffG * diffG
        sumB += diffB * diffB
        j += 3
      }
      i += 1
    }
    require(checkSum(sumR) && checkSum(sumG) & checkSum(sumB))
    stdR = math.sqrt(sumR / total).toFloat
    stdG = math.sqrt(sumG / total).toFloat
    stdB = math.sqrt(sumB / total).toFloat
  }

  override def transform(prev: Iterator[RGBImage]): Iterator[RGBImage] = {
    prev.map(img => {
      val content = img.content
      require(content.length % 3 == 0)
      var i = 0
      while (i < content.length) {
        content(i + 2) = ((content(i + 2) - meanR) / stdR).toFloat
        content(i + 1) = ((content(i + 1) - meanG) / stdG).toFloat
        content(i + 0) = ((content(i + 0) - meanB) / stdB).toFloat
        i += 3
      }
      img
    })
  }
}

class GreyImageCropper(cropWidth: Int, cropHeight: Int)
  extends Transformer[GreyImage, GreyImage] {

  import com.intel.analytics.sparkdl.utils.RandomGenerator.RNG

  private val buffer = new GreyImage(cropWidth, cropHeight)

  override def transform(prev: Iterator[GreyImage]): Iterator[GreyImage] = {
    prev.map(img => {
      val width = img.width()
      val height = img.height()
      val startW = RNG.uniform(0, width - cropWidth).toInt
      val startH = RNG.uniform(0, height - cropHeight).toInt
      val startIndex = startW + startH * width
      val frameLength = cropWidth * cropHeight
      val source = img.content
      val target = buffer.content
      var i = 0
      while (i < frameLength) {
        target(i) = source(startIndex + (i / cropWidth) * width +
          (i % cropWidth))
        i += 1
      }

      buffer.setLabel(img.label())
    })
  }
}

class RGBImageCropper(cropWidth: Int, cropHeight: Int)
  extends Transformer[RGBImage, RGBImage] {

  import com.intel.analytics.sparkdl.utils.RandomGenerator.RNG

  private val buffer = new RGBImage(cropWidth, cropHeight)

  override def transform(prev: Iterator[RGBImage]): Iterator[RGBImage] = {
    prev.map(img => {
      val width = img.width()
      val height = img.height()
      val startW = RNG.uniform(0, width - cropWidth).toInt
      val startH = RNG.uniform(0, height - cropHeight).toInt
      val startIndex = (startW + startH * width) * 3
      val frameLength = cropWidth * cropHeight
      val source = img.content
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
      buffer
    })
  }
}

class GreyImageToTensor(batchSize: Int) extends Transformer[GreyImage, (Tensor[Float],
  Tensor[Float])] {

  private def copyImage(img: GreyImage, storage: Array[Float], offset: Int): Unit = {
    val content = img.content
    val frameLength = img.width() * img.height()
    var j = 0
    while (j < frameLength) {
      storage(offset + j) = content(j)
      j += 1
    }
  }

  override def transform(prev: Iterator[GreyImage]): Iterator[(Tensor[Float], Tensor[Float])] = {
    new Iterator[(Tensor[Float], Tensor[Float])] {
      private val featureTensor: Tensor[Float] = Tensor[Float]()
      private val labelTensor: Tensor[Float] = Tensor[Float]()
      private var featureData: Array[Float] = null
      private var labelData: Array[Float] = null
      private var width = 0
      private var height = 0

      override def hasNext: Boolean = prev.hasNext

      override def next(): (Tensor[Float], Tensor[Float]) = {
        if (prev.hasNext) {
          var i = 0
          while (i < batchSize && prev.hasNext) {
            val img = prev.next()
            if(featureData == null) {
              featureData = new Array[Float](batchSize * img.height() * img.width())
              labelData = new Array[Float](batchSize)
              height = img.height()
              width = img.width()
            }
            copyImage(img, featureData, i * img.width() * img.height())
            labelData(i) = img.label()
            i += 1
          }
          if(labelTensor.nElement() != i) {
            featureTensor.set(Storage[Float](featureData),
              storageOffset = 1, sizes = Array(i, height, width))
            labelTensor.set(Storage[Float](labelData),
              storageOffset = 1, sizes = Array(i))
          }
          (featureTensor, labelTensor)
        } else {
          null
        }
      }
    }
  }
}

class RGBImageToTensor(batchSize: Int) extends Transformer[RGBImage, (Tensor[Float],
  Tensor[Float])] {

  private def copyImage(img: RGBImage, storage: Array[Float], offset: Int): Unit = {
    val content = img.content
    val frameLength = img.width() * img.height()
    require(content.length == frameLength * 3)
    var j = 0
    while (j < frameLength) {
      storage(offset + j) = content(j * 3)
      storage(offset + j + frameLength) = content(j * 3 + 1)
      storage(offset + j + frameLength * 2) = content(j * 3 + 2)
      j += 1
    }
  }

  override def transform(prev: Iterator[RGBImage]): Iterator[(Tensor[Float], Tensor[Float])] = {
    new Iterator[(Tensor[Float], Tensor[Float])] {
      private val featureTensor: Tensor[Float] = Tensor[Float]()
      private val labelTensor: Tensor[Float] = Tensor[Float]()
      private var featureData: Array[Float] = null
      private var labelData: Array[Float] = null
      private var width = 0
      private var height = 0

      override def hasNext: Boolean = prev.hasNext

      override def next(): (Tensor[Float], Tensor[Float]) = {
        if (prev.hasNext) {
          var i = 0
          while (i < batchSize && prev.hasNext) {
            val img = prev.next()
            if(featureData == null) {
              featureData = new Array[Float](batchSize * 3 * img.height() * img.width())
              labelData = new Array[Float](batchSize)
              height = img.height()
              width = img.width()
            }
            copyImage(img, featureData, i * img.width() * img.height() * 3)
            labelData(i) = img.label()
            i += 1
          }

          if(labelTensor.nElement() != i) {
            featureTensor.set(Storage[Float](featureData),
              storageOffset = 1, sizes = Array(i, 3, height, width))
            labelTensor.set(Storage[Float](labelData),
              storageOffset = 1, sizes = Array(i))
          }

          (featureTensor, labelTensor)
        } else {
          null
        }
      }
    }
  }
}
