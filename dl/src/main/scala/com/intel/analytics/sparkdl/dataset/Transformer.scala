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

import java.nio.file.Path
import java.util
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger

import com.fasterxml.jackson.databind.ser.std.StdJdkSerializers.AtomicIntegerSerializer
import com.intel.analytics.sparkdl.tensor.{Storage, Tensor}
import org.apache.commons.lang3.SerializationUtils

import scala.collection.Iterator
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContext, Future}
import scala.reflect.ClassTag

trait Transformer[A, B] extends Serializable {
  def transform(prev: Iterator[A]): Iterator[B]

  // scalastyle:off methodName
  def +[C](other: Transformer[B, C]): Transformer[A, C] = {
    new CombineTransformer(this, other)
  }

  // scalastyle:on methodName

  def cloneTransformer(): Transformer[A, B] = {
    SerializationUtils.clone(this)
  }
}

class CombineTransformer[A, B, C](first: Transformer[A, B], last: Transformer[B, C])
  extends Transformer[A, C] {
  override def transform(prev: Iterator[A]): Iterator[C] = {
    last.transform(first.transform(prev))
  }
}

class GreyImageNormalizer(dataSource: DataSource[GreyImage], samples: Int = -1)
  extends Transformer[GreyImage, GreyImage] {

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

object RGBImageNormalizer {
  def apply(meanR: Double, meanG: Double, meanB: Double,
    stdR: Double, stdG: Double, stdB: Double): RGBImageNormalizer = {

    new RGBImageNormalizer(meanR, meanG, meanB, stdR, stdG, stdB)
  }

  def apply(dataSource: LocalDataSource[RGBImage], samples: Int = -1): RGBImageNormalizer = {
    var sumR: Double = 0
    var sumG: Double = 0
    var sumB: Double = 0
    var total: Long = 0
    dataSource.shuffle()
    dataSource.reset()
    val totalCount = if (samples < 0) dataSource.total() else samples
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
      print(s"Mean: $i / $totalCount \r")
    }
    println()
    require(total > 0)
    val meanR = sumR / total
    val meanG = sumG / total
    val meanB = sumB / total
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
      print(s"Std: $i / $totalCount \r")
      i += 1
    }
    println()
    val stdR = math.sqrt(sumR / total)
    val stdG = math.sqrt(sumG / total)
    val stdB = math.sqrt(sumB / total)
    new RGBImageNormalizer(meanR, meanG, meanB, stdR, stdG, stdB)
  }
}

object ArrayByteToGreyImage {
  def apply(row: Int, col: Int): ArrayByteToGreyImage = new ArrayByteToGreyImage(row, col)
}

class ArrayByteToGreyImage(row: Int, col: Int)
  extends Transformer[(Float, Array[Byte]), GreyImage] {
  private val buffer = new GreyImage(row, col)

  override def transform(prev: Iterator[(Float, Array[Byte])]): Iterator[GreyImage] = {
    prev.map(rawData => {
      require(row * col == rawData._2.length)
      require(rawData._1 >= 1)
      buffer.setLabel(rawData._1).copy(rawData._2, 255.0f)
    })
  }
}

object ArrayByteToRGBImage {
  def apply(scale: Float = 255.0f): ArrayByteToRGBImage = new ArrayByteToRGBImage(scale)
}

class ArrayByteToRGBImage(scale: Float)
  extends Transformer[(Float, Array[Byte]), RGBImage] {
  private val buffer = new RGBImage()

  override def transform(prev: Iterator[(Float, Array[Byte])]): Iterator[RGBImage] = {
    prev.map(rawData => {
      buffer.copy(rawData._2, scale).setLabel(rawData._1)
    })
  }
}

object PathToRGBImage {
  def apply(scaleTo: Int): PathToRGBImage = new PathToRGBImage(scaleTo)
}

class PathToRGBImage(scaleTo: Int) extends Transformer[(Float, Path), RGBImage] {
  private val buffer = new RGBImage()

  override def transform(prev: Iterator[(Float, Path)]): Iterator[RGBImage] = {
    prev.map(data => {
      val imgData = RGBImage.readImage(data._2, scaleTo)
      val label = data._1
      buffer.copy(imgData).setLabel(label)
    })
  }
}

class RGBImageNormalizer(meanR: Double, meanG: Double, meanB: Double,
  stdR: Double, stdG: Double, stdB: Double)
  extends Transformer[RGBImage, RGBImage] {

  def getMean(): (Double, Double, Double) = (meanB, meanG, meanR)

  def getStd(): (Double, Double, Double) = (stdB, stdG, stdR)

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

object RGBImageCropper {
  def apply(cropWidth: Int, cropHeight: Int): RGBImageCropper =
    new RGBImageCropper(cropWidth, cropHeight)
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
      buffer.setLabel(img.label())
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
            if (featureData == null) {
              featureData = new Array[Float](batchSize * img.height() * img.width())
              labelData = new Array[Float](batchSize)
              height = img.height()
              width = img.width()
            }
            copyImage(img, featureData, i * img.width() * img.height())
            labelData(i) = img.label()
            i += 1
          }
          if (labelTensor.nElement() != i) {
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

class GreyImageToAETensor(batchSize: Int) extends Transformer[GreyImage, (Tensor[Float],
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
            if (featureData == null) {
              featureData = new Array[Float](batchSize * img.height() * img.width())
              labelData = new Array[Float](batchSize * img.height() * img.width())
              height = img.height()
              width = img.width()
            }
            copyImage(img, featureData, i * img.width() * img.height())
            labelData(i) = img.label()
            i += 1
          }
          if (labelTensor.nElement() != batchSize) {
            featureTensor.set(Storage[Float](featureData),
              storageOffset = 1, sizes = Array(batchSize, height * width))
            labelTensor.set(Storage[Float](featureData),
              storageOffset = 1, sizes = Array(batchSize, height * width))
          }
          (featureTensor, labelTensor)
        } else {
          null
        }
      }
    }
  }
}

object RGBImageToTensor {
  def apply(batchSize: Int): RGBImageToTensor = new RGBImageToTensor(batchSize)
}

class RGBImageToTensor(batchSize: Int) extends Transformer[RGBImage, (Tensor[Float],
  Tensor[Float])] {

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
            if (featureData == null) {
              featureData = new Array[Float](batchSize * 3 * img.height() * img.width())
              labelData = new Array[Float](batchSize)
              height = img.height()
              width = img.width()
            }
            img.copyTo(featureData, i * img.width() * img.height() * 3)
            labelData(i) = img.label()
            i += 1
          }

          if (labelTensor.nElement() != i) {
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

object MultiThreadRGBImageToSingleTensor {
  def apply[A: ClassTag](width: Int, height: Int, threadNum: Int, batchSize: Int,
    transformer: Transformer[A, RGBImage]): MultiThreadRGBImageToSingleTensor[A] = {
    new MultiThreadRGBImageToSingleTensor[A](width, height, threadNum, batchSize, transformer)
  }
}

class MultiThreadRGBImageToSingleTensor[A: ClassTag](width: Int, height: Int,
  threadNum: Int, batchSize: Int, transformer: Transformer[A, RGBImage])
  extends Transformer[A, (Tensor[Float], Tensor[Float])] {

  private val buffer = new Array[A](batchSize)
  private val transformers = (1 to batchSize).map(_ => transformer.cloneTransformer()).toArray
  private val frameLength = height * width
  private val featureData: Array[Float] = new Array[Float](batchSize * frameLength * 3)
  private val labelData: Array[Float] = new Array[Float](batchSize)
  private var pool: ExecutionContext = null
  private val featureTensor: Tensor[Float] = Tensor[Float]()
  private val labelTensor: Tensor[Float] = Tensor[Float]()

  def setPool(pool: ExecutionContext): this.type = {
    this.pool = pool
    this
  }

  def getPool(): ExecutionContext = {
    if (pool == null) {
      pool = new ExecutionContext {
        val threadPool = Executors.newFixedThreadPool(threadNum)

        def execute(runnable: Runnable) {
          threadPool.submit(runnable)
        }

        def reportFailure(t: Throwable) {}
      }
    }
    pool
  }


  override def transform(prev: Iterator[A]): Iterator[(Tensor[Float], Tensor[Float])] = {
    new Iterator[(Tensor[Float], Tensor[Float])] {
      override def hasNext: Boolean = prev.hasNext

      override def next(): (Tensor[Float], Tensor[Float]) = {
        var count = 0
        while (count < batchSize && prev.hasNext) {
          buffer(count) = prev.next()
          count += 1
        }

        (0 until count).map(i => Future {
          val img = transformers(i).transform(Iterator.single(buffer(i))).next()
          img.copyTo(featureData, i * frameLength * 3)
          labelData(i) = img.label()
        }(getPool())).foreach(Await.result(_, Duration.Inf))

        if (labelTensor.nElement() != count) {
          featureTensor.set(Storage[Float](featureData),
            storageOffset = 1, sizes = Array(count, 3, height, width))
          labelTensor.set(Storage[Float](labelData),
            storageOffset = 1, sizes = Array(count))
        }

        (featureTensor, labelTensor)
      }
    }
  }
}

object Identity {
  def apply[A](): Identity[A] = new Identity[A]()
}

class Identity[A] extends Transformer[A, A] {
  override def transform(prev: Iterator[A]): Iterator[A] = {
    prev
  }
}
