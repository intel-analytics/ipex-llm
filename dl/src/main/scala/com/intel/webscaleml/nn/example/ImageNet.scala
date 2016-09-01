package com.intel.webscaleml.nn.example

import java.nio.ByteBuffer
import java.nio.file.{Paths, Files}
import java.util
import java.util.Collections
import java.util.concurrent.{Executors, ExecutorService, LinkedBlockingQueue, BlockingQueue}

import com.intel.webscaleml.nn.tensor.{Storage, torch, Tensor}

object ImageNetUtils {

  def getLabels(labelsPath : String) : Map[String, Double] = {
    val labelsMapFile = scala.io.Source.fromFile(labelsPath)
    labelsMapFile.getLines().map(x => x.toString).toArray.map { line =>
      val parts = line.split(" ")
      (parts(0), parts(1).toDouble)
    }.toMap
  }

  def toTensorDouble(imgIter : Iterator[(Double, Array[Byte])], featureShape : Array[Int],
      labelShape : Array[Int], batchSize : Int, mean : (Double, Double, Double),
      std : (Double, Double, Double), input : Tensor[Double], target : Tensor[Double]) : Iterator[(Tensor[Double], Tensor[Double])] = {
    imgIter.grouped(batchSize).map(seq => {
      val size = seq.size
      require(input.nElement() >= size * featureShape.product)
      require(target.nElement() >= size)
      var i = 0
      val features = input.storage().array()
      val targets = target.storage().array()
      seq.foreach { case (label, data) =>
        cropDouble(data, input.size(3), input.size(4), mean, std, features, i * featureShape.product)
        targets(i) = label
        i += 1
      }
      (input, target)
    })
  }

  def toTensorFloat(imgIter : Iterator[(Float, Array[Byte])], featureShape : Array[Int],
               labelShape : Array[Int], batchSize : Int, mean : (Float, Float, Float),
               std : (Float, Float, Float), input : Tensor[Float], target : Tensor[Float]) : Iterator[(Tensor[Float], Tensor[Float])] = {
    imgIter.grouped(batchSize).map(seq => {
      val size = seq.size
      require(input.nElement() >= size * featureShape.product)
      require(target.nElement() >= size)
      var i = 0
      val features = input.storage().array()
      val targets = target.storage().array()
      seq.foreach { case (label, data) =>
        cropFloat(data, input.size(3), input.size(4), mean, std, features, i * featureShape.product)
        targets(i) = label
        i += 1
      }
      (input, target)
    })
  }

  def cropDouble(rawData : Array[Byte], cropWidth : Int, cropHeight : Int, mean : (Double, Double, Double),
           std : (Double, Double, Double), result : Array[Double], resultOffset : Int) : Unit = {
    val r = scala.util.Random
    r.nextBoolean()
    val buffer = ByteBuffer.wrap(rawData)
    val width = buffer.getInt
    val height = buffer.getInt

    val startW = r.nextInt(width - cropWidth)
    val startH = r.nextInt(height - cropHeight)
    val offset = 2 * 4
    val startIndex = startW + startH * width
    var i = 0
    val frameLength = cropWidth * cropHeight
    while(i < frameLength) {
      result(resultOffset + i) = ((rawData(offset + (startIndex + (i / cropWidth) * width +
        (i % cropWidth)) * 3 + 2) & 0xff) / 255.0 - mean._1) / std._1
      result(resultOffset + i + frameLength) = ((rawData(offset + (startIndex + (i / cropWidth) * width +
        (i % cropWidth)) * 3 + 1) & 0xff) / 255.0 - mean._2) / std._2
      result(resultOffset + i + frameLength * 2) = ((rawData(offset + (startIndex + (i / cropWidth) * width +
        (i % cropWidth)) * 3) & 0xff) / 255.0 - mean._3) / std._3
      i += 1
    }
  }

  def cropFloat(rawData : Array[Byte], cropWidth : Int, cropHeight : Int, mean : (Float, Float, Float),
           std : (Float, Float, Float), result : Array[Float], resultOffset : Int) : Unit = {
    val r = scala.util.Random
    r.nextBoolean()
    val buffer = ByteBuffer.wrap(rawData)
    val width = buffer.getInt
    val height = buffer.getInt

    val startW = r.nextInt(width - cropWidth)
    val startH = r.nextInt(height - cropHeight)
    val offset = 2 * 4
    val startIndex = startW + startH * width
    var i = 0
    val frameLength = cropWidth * cropHeight
    while(i < frameLength) {
      result(resultOffset + i) = ((rawData(offset + (startIndex + (i / cropWidth) * width +
        (i % cropWidth)) * 3 + 2) & 0xff) / 255.0f - mean._1) / std._1
      result(resultOffset + i + frameLength) = ((rawData(offset + (startIndex + (i / cropWidth) * width +
        (i % cropWidth)) * 3 + 1) & 0xff) / 255.0f - mean._2) / std._2
      result(resultOffset + i + frameLength * 2) = ((rawData(offset + (startIndex + (i / cropWidth) * width +
        (i % cropWidth)) * 3) & 0xff) / 255.0f - mean._3) / std._3
      i += 1
    }
  }

  def crop(rawData : Array[Byte], cropWidth : Int, cropHeight : Int, result : Array[Byte]) : Unit = {
    val r = scala.util.Random
    r.nextBoolean()
    val buffer = ByteBuffer.wrap(rawData)
    val width = buffer.getInt
    val height = buffer.getInt

    val startW = r.nextInt(width - cropWidth)
    val startH = r.nextInt(height - cropHeight)
    val offset = 2 * 4
    val startIndex = startW + startH * width
    var i = 0
    val frameLength = cropWidth * cropHeight
    while(i < frameLength) {
      result(i) = rawData(offset + (startIndex + (i / cropWidth) * width +
        (i % cropWidth)) * 3 + 2)
      result(i + frameLength) = rawData(offset + (startIndex + (i / cropWidth) * width +
        (i % cropWidth)) * 3 + 1)
      result(i + frameLength * 2) = rawData(offset + (startIndex + (i / cropWidth) * width +
        (i % cropWidth)) * 3)
      i += 1
    }
  }

  def normalizeDouble(rawData : Array[Byte], frameLength : Int, mean : (Double, Double, Double), std : (Double, Double, Double),
                result : Array[Double], resultOffset : Int) : Unit = {
    var i = 0
    while(i < frameLength) {
      result(resultOffset + i) = ((rawData(i) & 0xff) / 255.0 - mean._1) / std._1
      result(resultOffset + i + frameLength) = ((rawData(i + frameLength) & 0xff) / 255.0 - mean._2) / std._2
      result(resultOffset + i + frameLength * 2) = ((rawData(i + frameLength * 2) & 0xff) / 255.0 - mean._3) / std._3
      i += 1
    }
  }

  def normalizeFloat(rawData : Array[Byte], frameLength : Int, mean : (Float, Float, Float), std : (Float, Float, Float),
                result : Array[Float], resultOffset : Int) : Unit = {
    var i = 0
    while(i < frameLength) {
      result(resultOffset + i) = ((rawData(i) & 0xff) / 255.0f - mean._1) / std._1
      result(resultOffset + i + frameLength) = ((rawData(i + frameLength) & 0xff) / 255.0f - mean._2) / std._2
      result(resultOffset + i + frameLength * 2) = ((rawData(i + frameLength * 2) & 0xff) / 255.0f - mean._3) / std._3
      i += 1
    }
  }

  def computeMean(data : Array[Byte], dataOffset : Int) : (Double, Double, Double) = {
    require((data.length - dataOffset) % 3 == 0, "invalid data")
    var (sumR, sumG, sumB) = (0.0, 0.0, 0.0)
    var i = dataOffset
    while(i < data.length) {
      sumR += (data(i + 2) & 0xff) / 255.0
      sumG += (data(i + 1) & 0xff) / 255.0
      sumB += (data(i + 0) & 0xff) / 255.0
      i += 3
    }
    val number = (i - dataOffset) / 3.0
    (sumR / number, sumG / number, sumB / number)
  }

  def computeVar(data : Array[Byte], meanR : Double, meanG : Double, meanB : Double, dataOffset : Int)
      : (Double, Double, Double) = {
    require((data.length - dataOffset) % 3 == 0, "invalid data")
    var (sumR, sumG, sumB) = (0.0, 0.0, 0.0)
    var i = dataOffset
    while(i < data.length) {
      val r = ((data(i + 2) & 0xff) / 255.0 - meanR)
      val g = ((data(i + 1) & 0xff) / 255.0 - meanG)
      val b = ((data(i + 0) & 0xff) / 255.0 - meanB)
      sumR += r * r
      sumG += g * g
      sumB += b * b
      i += 3
    }
    val number = (i - dataOffset) / 3.0
    (math.sqrt(sumR / number), math.sqrt(sumG / number), math.sqrt(sumB / number))
  }
}

import javax.imageio.ImageIO
import java.awt._
import java.awt.image.BufferedImage
import java.awt.image.DataBufferByte
import java.io._
import java.nio.ByteBuffer
import java.nio.channels.Channels
import java.nio.file.Path


class Image(path : Path){
  val widthScale : Int = 256
  val heightScale : Int = 256
  val nChannels: Int = 3
  val cropWidth : Int = 224
  val cropHeight : Int = 224
  val dataOffset : Int = 8

  val label: String = path.getParent.getFileName.toString

  def load : Option[Array[Byte]] = {
    try {
      val fis = new FileInputStream(path.toString)
      val channel = fis.getChannel
      val byteArrayOutputStream = new ByteArrayOutputStream
      channel.transferTo(0, channel.size, Channels.newChannel(byteArrayOutputStream))
      val img = ImageIO.read(new ByteArrayInputStream(byteArrayOutputStream.toByteArray))
      if (img.getAlphaRaster != null) {
        throw new UnsupportedOperationException("Not support img with alpha channel")
      }
      var widthAfterScale = widthScale
      var heightAfterScale = heightScale
      if (img.getWidth < img.getHeight) {
        heightAfterScale = widthScale * img.getHeight / img.getWidth
      } else {
        widthAfterScale = heightScale * img.getWidth / img.getHeight
      }
      val scaledImage: java.awt.Image =
        img.getScaledInstance(widthAfterScale, heightAfterScale, java.awt.Image.SCALE_SMOOTH)
      val imageBuff: BufferedImage = new BufferedImage(widthAfterScale, heightAfterScale, BufferedImage.TYPE_3BYTE_BGR)
      imageBuff.getGraphics.drawImage(scaledImage, 0, 0, new Color(0, 0, 0), null)
      val pixels: Array[Byte] = (imageBuff.getRaster.getDataBuffer.asInstanceOf[DataBufferByte]).getData
      require(pixels.length % nChannels == 0)
      val buffer = new Array[Byte](dataOffset + pixels.length)
      val byteBuffer = ByteBuffer.wrap(buffer)
      byteBuffer.putInt(imageBuff.getWidth)
      byteBuffer.putInt(imageBuff.getHeight)
      System.arraycopy(pixels, 0, buffer, dataOffset, pixels.length)
      Some(buffer)
    } catch {
      case ex: Exception => {
        ex.printStackTrace
        System.err.println("Can't read file " + path)
        None
      }
    }
  }
}

class DataSets(pathName: String, classNum : Int, labelMap : Map[String, Double]) {
  private val imageList = new LinkedBlockingQueue[Image]()
  private var total = 0

  val path = Paths.get(pathName)
  val directoryStream = Files.newDirectoryStream(path)
  System.out.println("Start to read directories...")

  import scala.collection.JavaConversions._
  for (p <- directoryStream) {
    if(labelMap(p.getFileName.toString) < classNum + 0.5) {
      System.out.println("Read " + p.getFileName)
      val subDirectoryStream = Files.newDirectoryStream(p)
      for (image <- subDirectoryStream) {
        imageList.add(new Image(image))
        total += 1
      }
    }
  }
  System.out.println(s"Done($total)")

  def pull : Image = {
    imageList.take()
  }

  def post(img : Image) : Unit = {
    imageList.put(img)
  }

  def getTotal = total

  def shuffle : Unit = {
    val tmpList = new util.LinkedList[Image]()
    var img = imageList.poll()
    while(img != null) {
      tmpList.push(img)
      img = imageList.poll()
    }
    Collections.shuffle(tmpList)
    img = tmpList.poll()
    while(img != null) {
      imageList.put(img)
      img = tmpList.poll()
    }
  }
}

class Donkey(parallelism : Int, dataSet : DataSets) extends Iterator[(Array[Byte], Image)]{
  private val workers = Executors.newFixedThreadPool(parallelism)
  val loadedImages = new LinkedBlockingQueue[(Array[Byte], Image)](parallelism)
  @volatile var isStoped = false

  for(i <- 1 to parallelism) {
    workers.submit(new Runnable {
      override def run(): Unit = {
        println(s"donkey[$i] start to work")
        while(!isStoped) {
          val img = dataSet.pull
          val imgData = img.load
          if(imgData.isDefined) {
            loadedImages.put((imgData.get, img))
          }
        }
      }
    })
  }
  workers.shutdown()

  def pull : (Array[Byte], Image) = {
    loadedImages.take()
  }

  override def hasNext: Boolean = true

  override def next(): (Array[Byte], Image) = pull
}

