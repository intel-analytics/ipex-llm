package com.intel.analytics.dllib.lib.dataSet

import java.awt.Color
import java.awt.image.{BufferedImage, DataBufferByte}
import java.io.{ByteArrayInputStream, ByteArrayOutputStream, FileInputStream}
import java.nio.ByteBuffer
import java.nio.channels.Channels
import java.nio.file.{Files, Path, Paths}
import javax.imageio.ImageIO

import com.intel.analytics.dllib.lib.tensor.Tensor

import scala.util.Random

/**
  * Created by lzhang2 on 8/31/2016.
  */
object ImageUtils {
  def load(path : Path) : Option[Array[Byte]] = {
    val widthScale : Int = 256
    val heightScale : Int = 256
    val nChannels: Int = 3
    val cropWidth : Int = 224
    val cropHeight : Int = 224
    val dataOffset : Int = 8
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

  def loadPrimaryImage(pathName : String, labelsPath : String, fileSize : Int) : Array[(Double, Array[Byte])] = {
    val path = Paths.get(pathName)
    val directoryStream = Files.newDirectoryStream(path)
    System.out.println("Start to read directories...")

    //read labels
    val classNum = 1000 //???
    val labelsMapFile = scala.io.Source.fromFile(labelsPath)
    val labelMap = labelsMapFile.getLines().map(x => x.toString).toArray.map { line =>
      val parts = line.split(" ")
      (parts(0), parts(1).toDouble)
    }.toMap

    import scala.collection.JavaConversions._
    var res = new Array[(Double,Array[Byte])](fileSize)
    var labels = 0.0
    var total = 0
    for (p <- directoryStream) {
      if(labelMap(p.getFileName.toString) < classNum + 0.5) {
        System.out.println("Read " + p.getFileName)
        val subDirectoryStream = Files.newDirectoryStream(p)
        var labels = labelMap(p.getFileName.toString)
        for (image <- subDirectoryStream) {
          var imgData = load(image)
          if(imgData.isDefined) {
            res(total) = (labels,imgData.get)
          }
          total += 1
        }
      }
    }
    println("total:" + total)
    res
  }

  def loadBinaryFile(featureFile : String, labelFile : String): Array[(Double,Array[Byte])] = {
    val labelBuffer = ByteBuffer.wrap(Files.readAllBytes(Paths.get(labelFile)))
    val featureBuffer = ByteBuffer.wrap(Files.readAllBytes(Paths.get(featureFile)))
    val labelMagicNumber = labelBuffer.getInt()
    //require(labelMagicNumber == 2049) ???
    val featureMagicNumber = featureBuffer.getInt()
    //require(featureMagicNumber == 2051) ???
    val labelCount = labelBuffer.getInt()
    val featureCount = featureBuffer.getInt()
    require(labelCount == featureCount)
    val rowNum = featureBuffer.getInt()
    //require(rowNum == rowN)
    val colNum = featureBuffer.getInt()
    //require(colNum == colN)
    val result = new Array[(Double,Array[Byte])](featureCount)
    var i = 0
    while(i < featureCount) {
      val img = new Array[Byte]((rowNum * colNum + 1))
      var label = labelBuffer.get()
      var y = 0
      while(y < rowNum) {
        var x = 0
        while(x < colNum) {
          img(x + y * colNum) = featureBuffer.get()
          x += 1
        }
        y += 1
      }
      result(i) = (label, img)
      i += 1
    }
    result
  }

  def shuffle[T](data : Array[T]) = {
    var i = 0
    while(i < data.length) {
      val exchange = i + Random.nextInt(data.length - i)
      val tmp = data(exchange)
      data(exchange) = data(i)
      data(i) = tmp
      i += 1
    }
  }
}

object ComputeMeanStd {

  def rgbMeanStd(data : Array[(Double, Array[Byte])]): (Array[Double],Array[Double]) ={
    val samples = data.length
    var (meanR, meanG, meanB) = rgbMean(data)
    meanR /= samples
    meanG /= samples
    meanB /= samples
    var (stdR, stdG, stdB) = rgbVar(data,meanR, meanG, meanB)
    stdR /= samples
    stdG /= samples
    stdB /= samples
    (Array(meanR, meanG, meanB), Array(stdR, stdG, stdB))
  }

  def rgbMean(data : Array[(Double, Array[Byte])]) : (Double, Double, Double) = {
    data.map(d => computeSingleMean(d._2, 0)).
      reduce((a, b) => (a._1 + b._1, a._2 + b._2, a._3 + b._3))
  }

  def rgbVar(data : Array[(Double, Array[Byte])], meanR : Double, meanG : Double, meanB : Double)
  : (Double, Double, Double) = {
    data.map(d => computeSingleVar(d._2, meanR, meanG, meanB, 0)).
      reduce((a, b) => (a._1 + b._1, a._2 + b._2, a._3 + b._3))
  }

  def computeSingleMean(data : Array[Byte], dataOffset : Int) : (Double, Double, Double) = {
    //require((data.length - dataOffset) % 3 == 0, "invalid data") //???
    var (sumR, sumG, sumB) = (0.0, 0.0, 0.0)
    var i = dataOffset
    while(i < (data.length-2)) {
      sumR += (data(i + 2) & 0xff) / 255.0
      sumG += (data(i + 1) & 0xff) / 255.0
      sumB += (data(i + 0) & 0xff) / 255.0
      i += 3
    }
    val number = (i - dataOffset) / 3.0
    (sumR / number, sumG / number, sumB / number)
  }

  def computeSingleVar(data : Array[Byte], meanR : Double, meanG : Double, meanB : Double, dataOffset : Int)
  : (Double, Double, Double) = {
    //require((data.length - dataOffset) % 3 == 0, "invalid data") //???
    var (sumR, sumG, sumB) = (0.0, 0.0, 0.0)
    var i = dataOffset
    while(i < (data.length-2)) {
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

  def grayMeanStd(data : Array[(Double,Array[Byte])]) : (Array[Double], Array[Double]) = {
    var imgSize = 0
    var i = 0
    var length = data.length
    var count = data.length
    var total = 0.0
    while(i < length){
      var d= data(i)
      var j = 0
      var sum = 0.0
      var img = d._2
      imgSize = img.length
      while(j < img.length) {
        sum += (img(j) & 0xff) / 255.0
        j += 1
      }
      i += 1
      total += sum
    }
    val mean = total / (count * imgSize)
    var stdTotal = 0.0
    var m = 0
    while(m < length){
      var d= data(m)
      var j = 0
      var sum = 0.0
      var img = d._2
      imgSize = img.length
      while(j < img.length) {
        val s = (img(j) & 0xff) / 255.0 - mean
        sum += s * s
        j += 1
      }
      m += 1
      stdTotal += sum
    }

    val std = math.sqrt(stdTotal / (count * imgSize))
    (Array(mean), Array(std))
  }
}

object toTensor {

  def toTensorRGB(imageData : Array[(Double, Array[Byte])], featureShape : Array[Int],
                     labelShape : Array[Int], batchSize : Int, mean : (Double, Double, Double),
                     std : (Double, Double, Double), input : Tensor[Double], target : Tensor[Double]) : Iterator[(Tensor[Double], Tensor[Double])] = {
    var imgIter = imageData.toIterator
    imgIter.grouped(batchSize).map(seq => {
      val size = seq.size
      require(input.nElement() >= size * featureShape.product)
      require(target.nElement() >= size)
      var i = 0
      val features = input.storage().array()
      val targets = target.storage().array()
      seq.foreach { case (label, data) =>
        cropRGB(data, input.size(3), input.size(4), mean, std, features, i * featureShape.product)
        targets(i) = label
        i += 1
      }
      (input, target)
    })
  }

  def cropRGB(rawData : Array[Byte], cropWidth : Int, cropHeight : Int, mean : (Double, Double, Double),
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

  def normalizeRGB(rawData : Array[Byte], frameLength : Int, mean : (Double, Double, Double), std : (Double, Double, Double),
                      result : Array[Double], resultOffset : Int) : Unit = {
    var i = 0
    while(i < frameLength) {
      result(resultOffset + i) = ((rawData(i) & 0xff) / 255.0 - mean._1) / std._1
      result(resultOffset + i + frameLength) = ((rawData(i + frameLength) & 0xff) / 255.0 - mean._2) / std._2
      result(resultOffset + i + frameLength * 2) = ((rawData(i + frameLength * 2) & 0xff) / 255.0 - mean._3) / std._3
      i += 1
    }
  }

  def toTensorForGray(imageData : Array[(Double, Array[Byte])], featureShape : Array[Int],
                      labelShape : Array[Int], batchSize : Int, mean : Double, std : Double, input : Tensor[Double], target : Tensor[Double]) : Iterator[(Tensor[Double], Tensor[Double])] = {
    var imgIter = imageData.toIterator
    imgIter.grouped(batchSize).map(seq => {
      val size = seq.size
      //require(input.nElement() >= size * featureShapeSize)
      //require(target.nElement() >= size)
      var i = 0
      val features = input.storage().array()
      val targets = target.storage().array()
      seq.foreach { case (label, data) =>
        normalizeGray(data, data.size, mean, std, features, i * featureShape.product)
        targets(i) = label
        i += 1
      }
      (input, target)
    })
  }

  def normalizeGray(rawData : Array[Byte], frameLength : Int, mean : Double, std : Double, result : Array[Double], resultOffset : Int) : Unit = {
    var i = 0
    while(i < frameLength) {
      result(resultOffset + i) = ((rawData(i) & 0xff) / 255.0 - mean) / std
      i += 1
    }
  }

  def cropGray(rawData : Array[Byte], cropWidth : Int, cropHeight : Int, mean : Double, std : Double, result : Array[Double], resultOffset : Int) : Unit = {
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
        (i % cropWidth)) + 2) & 0xff) / 255.0 - mean) / std
      i += 1
    }
  }

}
