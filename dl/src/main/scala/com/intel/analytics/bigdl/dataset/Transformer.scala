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
package com.intel.analytics.bigdl.dataset


import java.nio.ByteBuffer
import java.nio.file.Path
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.{Executors, ThreadFactory}

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.RandomGenerator
import org.apache.commons.lang3.SerializationUtils
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{Path => hadoopPath}
import org.apache.hadoop.io.SequenceFile.Reader
import org.apache.hadoop.io.{SequenceFile, Text}

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

/**
 * This class compress or expend the image to a specific size
 *
 * @param width the width of an image to be converted
 * @param height the height of an image to be converted
 */
class Scaler(width: Int, height: Int) extends Transformer[LabeledImage, LabeledImage] {
  override def transform(prev: Iterator[LabeledImage]): Iterator[LabeledImage] = {
    prev.map(img => {
      ImageUtils.scale(img, width, height)
    })
  }
}

class HFlip(threshold: Double) extends Transformer[LabeledImage, LabeledImage] {
  override def transform(prev: Iterator[LabeledImage]): Iterator[LabeledImage] = {
    prev.map(img => {
      if (RandomGenerator.RNG.uniform(0, 1) >= threshold) {
        ImageUtils.hFlip(img)
      } else {
        img
      }
    })
  }
}

class ImageCropper(cropWidth: Int, cropHeight: Int, numChannels: Int)
  extends Transformer[LabeledImage, LabeledImage] {
  val buffer = new LabeledImage(cropWidth, cropHeight, numChannels)

  import com.intel.analytics.bigdl.utils.RandomGenerator.RNG

  override def transform(prev: Iterator[LabeledImage]): Iterator[LabeledImage] = {
    prev.map(img => {
      val width = img.width
      val height = img.height
      val numChannels = img.nChannels
      val startW = RNG.uniform(0, width - cropWidth).toInt
      val startH = RNG.uniform(0, height - cropHeight).toInt
      val startIndex = (startW + startH * width) * numChannels
      val frameLength = cropWidth * cropHeight
      val source = img.content
      val target = buffer.content
      var i = 0
      while (i < frameLength) {
        var j = 0
        while (j < numChannels) {
          target(i * numChannels + j) =
            source(startIndex + ((i / cropWidth) * width + (i % cropWidth)) * numChannels + j)
          j += 1
        }
        i += 1
      }
      buffer.setLabel(img.label)
    })
  }
}

case class Identity[A: ClassTag]() extends Transformer[A, A] {
  override def transform(prev: Iterator[A]): Iterator[A] = {
    prev
  }
}

object PathToGreyImage {
  def apply(scaleTo: Int): PathToImage = new PathToImage(1, scaleTo)
}

object PathToRGBImage {
  def apply(scaleTo: Int): PathToImage = new PathToImage(3, scaleTo)
}

class PathToImage(numChannels: Int, scaleTo: Int) extends Transformer[(Float, Path), LabeledImage] {
  private val buffer = new LabeledImage(0, 0, numChannels)

  override def transform(x: Iterator[(Float, Path)]): Iterator[LabeledImage] = {
    x.map(data => {
      val imgData = ImageUtils.readBGRImage(data._2, scaleTo)
      val label = data._1
      buffer.copy(imgData, numChannels).setLabel(label)
    })
  }
}

object ArrayByteToGreyImage {
  def apply(scale: Float = 1.0f): ArrayByteToImage = new ArrayByteToImage(1, scale)
}

object ArrayByteToRGBImage {
  def apply(scale: Float = 255.0f): ArrayByteToImage = new ArrayByteToImage(3, scale)
}

class ArrayByteToImage(numChannels: Int, scale: Float)
  extends Transformer[(Float, Array[Byte]), LabeledImage] {
  private val buffer = new LabeledImage(0, 0, 0)

  override def transform(prev: Iterator[(Float, Array[Byte])]): Iterator[LabeledImage] = {
    prev.map(rawData => {
      buffer.copy(rawData._2, numChannels, scale).setLabel(rawData._1)
    })
  }
}
//
//object RGBImageNormalizer {
//  def apply(meanR: Double, meanG: Double, meanB: Double,
//    stdR: Double, stdG: Double, stdB: Double): RGBImageNormalizer = {
//
//    new RGBImageNormalizer(meanR, meanG, meanB, stdR, stdG, stdB)
//  }
//
//  def apply(dataSource: LocalDataSource[LabeledImage], samples: Int = Int.MaxValue)
//  : RGBImageNormalizer = {
//    var sumR: Double = 0
//    var sumG: Double = 0
//    var sumB: Double = 0
//    var total: Long = 0
//    dataSource.shuffle()
//    dataSource.reset()
//    val totalCount = if (samples < 0) dataSource.total() else samples
//    var i = 0
//    while (i < math.min(samples, dataSource.total())) {
//      val content = dataSource.next().content
//      require(content.length % 3 == 0)
//      var j = 0
//      while (j < content.length) {
//        sumR += content(j + 2)
//        sumG += content(j + 1)
//        sumB += content(j + 0)
//        total += 1
//        j += 3
//      }
//      i += 1
//      print(s"Mean: $i / $totalCount \r")
//    }
//    println()
//    require(total > 0)
//    val meanR = sumR / total
//    val meanG = sumG / total
//    val meanB = sumB / total
//    sumR = 0
//    sumG = 0
//    sumB = 0
//    i = 0
//    dataSource.reset()
//    while (i < math.min(samples, dataSource.total())) {
//      val content = dataSource.next().content
//      var j = 0
//      while (j < content.length) {
//        val diffR = content(j + 2) - meanR
//        val diffG = content(j + 1) - meanG
//        val diffB = content(j + 0) - meanB
//        sumR += diffR * diffR
//        sumG += diffG * diffG
//        sumB += diffB * diffB
//        j += 3
//      }
//      print(s"Std: $i / $totalCount \r")
//      i += 1
//    }
//    println()
//    val stdR = math.sqrt(sumR / total)
//    val stdG = math.sqrt(sumG / total)
//    val stdB = math.sqrt(sumB / total)
//    new RGBImageNormalizer(meanR, meanG, meanB, stdR, stdG, stdB)
//  }
//}
//
//class RGBImageNormalizer(meanR: Double, meanG: Double, meanB: Double,
//  stdR: Double, stdG: Double, stdB: Double)
//  extends Transformer[LabeledImage, LabeledImage] {
//
//  def getMean(): (Double, Double, Double) = (meanB, meanG, meanR)
//
//  def getStd(): (Double, Double, Double) = (stdB, stdG, stdR)
//
//  override def transform(prev: Iterator[LabeledImage]): Iterator[LabeledImage] = {
//    prev.map(img => {
//      val content = img.content
//      require(content.length % 3 == 0)
//      var i = 0
//      while (i < content.length) {
//        content(i + 2) = ((content(i + 2) - meanR) / stdR).toFloat
//        content(i + 1) = ((content(i + 1) - meanG) / stdG).toFloat
//        content(i + 0) = ((content(i + 0) - meanB) / stdB).toFloat
//        i += 3
//      }
//      img
//    })
//  }
//}

class ImageNormalizer(
  var means: Array[Double],
  var stds: Array[Double])
  extends Transformer[LabeledImage, LabeledImage] {

  def this(
    dataSource: LocalDataSource[LabeledImage],
    samples: Int = Int.MaxValue) = {
    this(null, null)
    val p = getMeansAndStds(dataSource, samples)
    means = p._1
    stds = p._2
  }

  def getMean: Array[Double] = means

  def getStd: Array[Double] = stds

  override def transform(prev: Iterator[LabeledImage]): Iterator[LabeledImage] = {
    prev.map(img => {
     ImageUtils.normalize(img, means, stds)
    })
  }

  def getMeansAndStds(
    dataSource: LocalDataSource[LabeledImage],
    samples: Int = Int.MaxValue): (Array[Double], Array[Double]) = {
      var sums: Array[Double] = null
      var total: Long = 0
      dataSource.shuffle()
      dataSource.reset()
      val totalCount = if (samples < 0) dataSource.total() else samples
      var i = 0
      while (i < math.min(samples, dataSource.total())) {
        val img = dataSource.next()
        if (sums == null) {
          sums = new Array(img.nChannels)
        }
        val content = img.content
        require(content.length % img.nChannels == 0)
        var j = 0
        while (j < content.length) {
          var c = 0
          while (c < img.nChannels) {
            sums(c) += content(j + c)
            c += 1
          }
          total += 1
          j += img.nChannels
        }
        i += 1
        print(s"Mean: $i / $totalCount \r")
      }
      println()
      require(total > 0)
      val means = sums.map(_ / total )
      sums = sums.map(_ => 0.0)

      i = 0
      dataSource.reset()
      while (i < math.min(samples, dataSource.total())) {
        val img = dataSource.next()
        val content = img.content
        var j = 0
        while (j < content.length) {
          var c = 0
          while (c < img.nChannels) {
            val diff = content(j + c) - means(c)
            sums(c) += diff * diff
            c += 1
          }
          j += img.nChannels
        }
        print(s"Std: $i / $totalCount \r")
        i += 1
      }
      println()
      val stds = sums.map(sum => math.sqrt(sum / total))
      (means, stds)
  }
}

object SeqFileToArrayByte {
  def apply(): SeqFileToArrayByte = new SeqFileToArrayByte()
}

class SeqFileToArrayByte extends Transformer[Path, (Float, Array[Byte])] {

  import org.apache.hadoop.fs.{Path => hPath}

  @transient
  private var key: Text = null

  @transient
  private var value: Text = null

  @transient
  private var reader: SequenceFile.Reader = null

  @transient
  private var oneRecordBuffer: (Float, Array[Byte]) = null

  override def transform(prev: Iterator[Path]): Iterator[(Float, Array[Byte])] = {
    new Iterator[(Float, Array[Byte])] {
      override def next(): (Float, Array[Byte]) = {
        if (oneRecordBuffer != null) {
          val res = oneRecordBuffer
          oneRecordBuffer = null
          return res
        }

        if (key == null) {
          key = new Text()
        }
        if (value == null) {
          value = new Text
        }
        if (reader == null || !reader.next(key, value)) {
          if (reader != null) {
            reader.close()
          }

          reader = new SequenceFile.Reader(new Configuration,
            Reader.file(new hPath(prev.next().toAbsolutePath.toString)))
          reader.next(key, value)
        }

        (key.toString.toFloat, value.copyBytes())
      }

      override def hasNext: Boolean = {
        if (oneRecordBuffer != null) {
          true
        } else if (reader == null) {
          prev.hasNext
        } else {
          if (reader.next(key, value)) {
            oneRecordBuffer = (key.toString.toFloat, value.copyBytes())
            true
          } else {
            prev.hasNext
          }
        }
      }
    }
  }
}


class ImageToTensor(batchSize: Int) extends Transformer[LabeledImage, (Tensor[Float],
  Tensor[Float])] {

  override def transform(prev: Iterator[LabeledImage]): Iterator[(Tensor[Float], Tensor[Float])] =
    new Iterator[(Tensor[Float], Tensor[Float])] {
      private val featureTensor: Tensor[Float] = Tensor[Float]()
      private val labelTensor: Tensor[Float] = Tensor[Float]()
      private var featureData: Array[Float] = null
      private var labelData: Array[Float] = null
      private var width = 0
      private var height = 0
      private var numChannels = 0

      override def hasNext: Boolean = prev.hasNext

      override def next(): (Tensor[Float], Tensor[Float]) = {
        if (prev.hasNext) {
          var i = 0
          while (i < batchSize && prev.hasNext) {
            val img = prev.next()
            if (featureData == null) {
              featureData = new Array[Float](batchSize * img.nChannels * img.height * img.width)
              labelData = new Array[Float](batchSize)
              height = img.height
              width = img.width
              numChannels = img.nChannels
            }
            img.copyTo(featureData, i * img.width * img.height * img.nChannels)
            labelData(i) = img.label
            i += 1
          }

          if (labelTensor.nElement() != i) {
            featureTensor.set(Storage[Float](featureData),
              storageOffset = 1, sizes = Array(i, numChannels, height, width))
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

class ImageToSequentialFile(blockSize: Int, baseFileName: Path) extends
  Transformer[LabeledImage, String] {
  private val conf: Configuration = new Configuration
  private var index = 0
  private val preBuffer: ByteBuffer = ByteBuffer.allocate(4 * 2)

  override def transform(prev: Iterator[LabeledImage]): Iterator[String] = {
    new Iterator[String] {
      override def hasNext: Boolean = prev.hasNext

      override def next(): String = {
        val fileName = baseFileName + s"_$index"
        val path = new hadoopPath(fileName)
        val writer = SequenceFile.createWriter(conf, SequenceFile.Writer.file(path),
          SequenceFile.Writer.keyClass(classOf[Text]),
          SequenceFile.Writer.valueClass(classOf[Text]))
        var i = 0
        while (i < blockSize && prev.hasNext) {
          val image = prev.next()
          preBuffer.putInt(image.width)
          preBuffer.putInt(image.height)
          val imageByteData = ImageUtils.convertToByte(image)
          val data: Array[Byte] = new Array[Byte](preBuffer.capacity + imageByteData.length)
          System.arraycopy(preBuffer.array, 0, data, 0, preBuffer.capacity)
          System.arraycopy(imageByteData, 0, data, preBuffer.capacity, imageByteData.length)
          preBuffer.clear
          val imageKey = s"${image.label toInt}"
          writer.append(new Text(imageKey), new Text(data))
          i += 1
        }
        writer.close()
        index += 1
        fileName
      }
    }
  }
}

class MultiThreadImageToSingleTensor[A: ClassTag](
  width: Int,
  height: Int,
  numChannels: Int,
  threadNum: Int,
  batchSize: Int,
  transformer: Transformer[A, LabeledImage])
  extends Transformer[A, (Tensor[Float], Tensor[Float])] {

  private val transformers = (1 to batchSize).map(_ => transformer.cloneTransformer()).toArray
  private val frameLength = height * width
  private val featureData: Array[Float] = new Array[Float](batchSize * frameLength * numChannels)
  private val labelData: Array[Float] = new Array[Float](batchSize)
  private var pool: ExecutionContext = null
  private val featureTensor: Tensor[Float] = Tensor[Float]()
  private val labelTensor: Tensor[Float] = Tensor[Float]()

  def setPool(pool: ExecutionContext): this.type = {
    this.pool = pool
    this
  }

  def getPool: ExecutionContext = {
    if (pool == null) {
      pool = new ExecutionContext {
        val threadPool = Executors.newFixedThreadPool(threadNum,
          new ThreadFactory {
            override def newThread(r: Runnable): Thread = {
              val t = new Thread(r)
              t.setDaemon(true)
              t
            }
          })

        def execute(runnable: Runnable) {
          threadPool.submit(runnable)
        }

        def reportFailure(t: Throwable) {}
      }
    }
    pool
  }


  override def transform(prev: Iterator[A]): Iterator[(Tensor[Float], Tensor[Float])] = {
    val iterators = transformers.map(_.transform(prev))

    new Iterator[(Tensor[Float], Tensor[Float])] {
      override def hasNext: Boolean = {
        iterators.map(_.hasNext).reduce(_ || _)
      }

      override def next(): (Tensor[Float], Tensor[Float]) = {
        val count = new AtomicInteger(0)
        (0 until batchSize).map(tid => Future {
          if (iterators(tid).hasNext) {
            val position = count.getAndIncrement()
            val img = iterators(tid).next()
            img.copyTo(featureData, position * frameLength * numChannels)
            labelData(position) = img.label
          }
        }(getPool)).foreach(Await.result(_, Duration.Inf))

        if (labelTensor.nElement() != count.get()) {
          featureTensor.set(Storage[Float](featureData),
            storageOffset = 1, sizes = Array(count.get(), numChannels, height, width))
          labelTensor.set(Storage[Float](labelData),
            storageOffset = 1, sizes = Array(count.get()))
        }

        (featureTensor, labelTensor)
      }
    }
  }
}

