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

import org.apache.hadoop.fs.{Path => hadoopPath}
import java.nio.file.Path
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.{Executors, ThreadFactory}


import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.RandomGenerator
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.apache.commons.lang3.SerializationUtils
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.SequenceFile.Reader
import org.apache.hadoop.io.{SequenceFile, Text}

import scala.collection.Iterator
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContext, Future}
import scala.reflect.ClassTag
import scala.util.Random

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

class GreyImageNormalizer(dataSource: DataSource[GreyImage], samples: Int = Int.MaxValue)
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
    while (i < math.min(samples, dataSource.total())) {
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
    while (i < math.min(samples, dataSource.total())) {
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

  def apply(dataSource: LocalDataSource[RGBImage], samples: Int = Int.MaxValue)
  : RGBImageNormalizer = {
    var sumR: Double = 0
    var sumG: Double = 0
    var sumB: Double = 0
    var total: Long = 0
    dataSource.shuffle()
    dataSource.reset()
    val totalCount = if (samples < 0) dataSource.total() else samples
    var i = 0
    while (i < math.min(samples, dataSource.total())) {
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
    while (i < math.min(samples, dataSource.total())) {
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
            return true
          } else {
            prev.hasNext
          }
        }
      }
    }
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

  import com.intel.analytics.bigdl.utils.RandomGenerator.RNG

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

object RGBImageRandomCropper {
  def apply(cropWidth: Int, cropHeight: Int, padding: Int): RGBImageRandomCropper =
    new RGBImageRandomCropper(cropHeight, cropWidth, padding)
}

class RGBImageRandomCropper(cropHeight: Int, cropWidth: Int, padding: Int)
  extends Transformer[RGBImage, RGBImage] {
  import com.intel.analytics.bigdl.utils.RandomGenerator.RNG

  private val buffer = new RGBImage(cropWidth, cropHeight)

  override def transform(prev: Iterator[RGBImage]): Iterator[RGBImage] = {
    prev.map(img => {
      if (padding > 0) {
        val widthTmp = img.width()
        val heightTmp = img.height()
        val sourceTmp = img.content
        val padWidth = widthTmp + 2*padding
        val padHeight = heightTmp + 2*padding
        val temp = new RGBImage(padWidth, padHeight)
        val tempBuffer = temp.content
        val startIndex = (padding+1 + (padding+1)*padWidth)*3
        val frameLength = widthTmp * heightTmp
        var i = 0
        while (i < frameLength) {
          tempBuffer(startIndex + ((i/widthTmp)*padWidth + (i%widthTmp))*3 + 2) = sourceTmp(i*3+2)
          tempBuffer(startIndex + ((i/widthTmp)*padWidth + (i%widthTmp))*3 + 1) = sourceTmp(i*3+1)
          tempBuffer(startIndex + ((i/widthTmp)*padWidth + (i%widthTmp))*3) = sourceTmp(i*3)
          i += 1
        }
        temp.setLabel(img.label())
        //img.copy(temp)
      }

      val width = img.width()
      val height = img.height()
      val source = img.content

      val startW = RNG.uniform(0, width - cropWidth).toInt
      val startH = RNG.uniform(0, height - cropHeight).toInt
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

  import com.intel.analytics.bigdl.utils.RandomGenerator.RNG

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

object ColorJitter {
  def apply(): ColorJitter = {
    new ColorJitter()
  }
}
class ColorJitter extends Transformer[RGBImage, RGBImage] {
  val bcsParameters = Map("brightness" -> 0.4f, "contrast" -> 0.4f, "saturation" -> 0.4f)

  def grayScale(dst: Array[Float], img: Array[Float]): Array[Float] = {
    var i = 0
    while (i < img.length) {
      dst(i) = img(i)*0.299f + img(i+1)*0.587f + img(i+2)*0.114f
      dst(i+1) = dst(i)
      dst(i+2) = dst(i)
      i += 3
    }
    dst
  }

  def blend(img1: Array[Float], img2: Array[Float], alpha: Float) =
    (img1 zip img2) map {case (a,b) => a + (1-alpha)*b }

  def saturation(variance: Float)(input: Array[Float]) = {
    val gs = new Array[Float](input.length)
    grayScale(gs, input)
    val alpha = 1.0f + RNG.uniform(-variance, variance).toFloat
    blend(input, gs, alpha)
    input
  }

  def brightness(variance: Float)(input: Array[Float])= {
    val gs = new Array[Float](input.length)
    val alpha = 1.0f + RNG.uniform(-variance, variance).toFloat
    blend(input, gs, alpha)
    input
  }

  def contrast(variance: Float)(input: Array[Float]) = {
    val gs = new Array[Float](input.length)
    grayScale(gs, input)
    val mean = gs.sum / gs.length
    gs.foreach( _ => mean)
    val alpha = 1.0f + RNG.uniform(-variance, variance).toFloat
    blend(input, gs, alpha)
    input
  }

  val ts = Map(
    "brightness" -> {brightness(bcsParameters.get("brightness").get)(_)},
    "contrast"   -> {contrast(bcsParameters.get("contrast").get)(_)},
    "saturation" -> {saturation(bcsParameters.get("saturation").get)(_)}
  )

  def randomOrder(input: Array[Float]): Unit = {
    val randOrder = Random.shuffle(List("brightness", "contrast", "saturation"))
    randOrder.map( x => ts(x))
  }

  override def transform(prev: Iterator[RGBImage]): Iterator[RGBImage] = {
    prev.map(img => {
      val content = img.content
      require(content.length % 3 == 0)
      randomOrder(content)
      img
    })
  }
}

object Lighting {
  def apply(): Lighting = {
    new Lighting()
  }
}

class Lighting extends Transformer[RGBImage, RGBImage] {
  val alphastd = 0.1f
  val eigval = Tensor[Float](Storage(Array( 0.2175f, 0.0188f, 0.0045f )), 1, Array(3))
  val eigvec = Tensor[Float](Storage(Array( -0.5675f,  0.7192f,  0.4009f,
    -0.5808f, -0.0045f, -0.8140f,
    -0.5836f, -0.6948f,  0.4203f)), 1, Array(3, 3))

  def lighting(input: Array[Float]): Unit = {
    if (alphastd != 0) {
      val alpha = Tensor[Float](3).apply1(_ => RNG.uniform(0, alphastd).toFloat)
      val rgb = eigvec.clone
        .cmul(alpha.view(1, 3).expand(Array(3, 3)))
        .cmul(eigval.view(1, 3).expand(Array(3, 3)))
        .sum(2).squeeze
      var i = 0
      while (i < input.length) {
        input(i) = input(i) + rgb.storage().array()(0)
        input(i+1) = input(i+1) + rgb.storage().array()(1)
        input(i+2) = input(i+2) + rgb.storage().array()(2)
        i += 3
      }
    }
  }

  override def transform(prev: Iterator[RGBImage]): Iterator[RGBImage] = {
    prev.map(img => {
      lighting(img.content)
      img
    })
  }
}

object HFlip {
  def apply(threshold: Double): HFlip = {
    new HFlip(threshold)
  }
}

class HFlip(threshold: Double) extends Transformer[RGBImage, RGBImage] {
  override def transform(prev: Iterator[RGBImage]): Iterator[RGBImage] = {
    prev.map(img => {
      if (RandomGenerator.RNG.uniform(0, 1) >= threshold) {
        img.hflip()
      } else {
        img
      }
    })
  }
}

object RGBImageToSequentialFile {
  def apply(blockSize: Int, baseFileName: Path): RGBImageToSequentialFile = {
    new RGBImageToSequentialFile(blockSize, baseFileName)
  }
}

class RGBImageToSequentialFile(blockSize: Int, baseFileName: Path) extends
  Transformer[RGBImage, String] {
  private val conf: Configuration = new Configuration
  private var index = 0
  private val preBuffer: ByteBuffer = ByteBuffer.allocate(4 * 2)

  override def transform(prev: Iterator[RGBImage]): Iterator[String] = {
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
          preBuffer.putInt(image.width())
          preBuffer.putInt(image.height())
          val imageByteData = image.convertToByte()
          val data: Array[Byte] = new Array[Byte](preBuffer.capacity + imageByteData.length)
          System.arraycopy(preBuffer.array, 0, data, 0, preBuffer.capacity)
          System.arraycopy(imageByteData, 0, data, preBuffer.capacity, imageByteData.length)
          preBuffer.clear
          val imageKey = s"${image.label().toInt}"
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

object MultiThreadRGBImageToSingleTensor {
  def apply[A: ClassTag](width: Int, height: Int, threadNum: Int, batchSize: Int,
    transformer: Transformer[A, RGBImage]): MultiThreadRGBImageToSingleTensor[A] = {
    new MultiThreadRGBImageToSingleTensor[A](width, height, threadNum, batchSize, transformer)
  }
}

class MultiThreadRGBImageToSingleTensor[A: ClassTag](width: Int, height: Int,
  threadNum: Int, batchSize: Int, transformer: Transformer[A, RGBImage])
  extends Transformer[A, (Tensor[Float], Tensor[Float])] {

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
            img.copyTo(featureData, position * frameLength * 3)
            labelData(position) = img.label()
          }
        }(getPool())).foreach(Await.result(_, Duration.Inf))

        if (labelTensor.nElement() != count.get()) {
          featureTensor.set(Storage[Float](featureData),
            storageOffset = 1, sizes = Array(count.get(), 3, height, width))
          labelTensor.set(Storage[Float](labelData),
            storageOffset = 1, sizes = Array(count.get()))
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
