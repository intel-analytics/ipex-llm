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

import java.awt.Color
import java.awt.color.ColorSpace
import java.awt.image.{BufferedImage, DataBufferByte}
import java.io.{ByteArrayInputStream, ByteArrayOutputStream, File, FileInputStream}
import java.nio.ByteBuffer
import java.nio.channels.Channels
import java.nio.file.{Files, Path, Paths}
import javax.imageio.ImageIO

import com.intel.analytics.sparkdl.utils.RandomGenerator

abstract class Image(protected var data: Array[Float], protected var _width: Int,
  protected var _height: Int, protected var _label: Float) {

  def width(): Int = _width

  def height(): Int = _height

  def content: Array[Float] = data

  def label(): Float = _label

  def setLabel(label: Float): this.type = {
    this._label = label
    this
  }
}

class GreyImage(d: Array[Float], w: Int, h: Int, l: Float) extends Image(d, w, h, l) {
  def this(_width: Int, _height: Int) =
    this(new Array[Float](_width * _height), _width, _height, 0.0f)

  def this() = this(new Array[Float](0), 0, 0, 0)

  def copy(source: Array[Byte], scale: Float = 1.0f, offset: Int = 0): this.type = {
    require(data.length + offset <= source.length)
    var i = 0
    while (i < data.length) {
      data(i) = (source(i + offset) & 0xff) / scale
      i += 1
    }
    this
  }

  def copy(other: GreyImage): GreyImage = {
    this._width = other._width
    this._height = other._height
    this._label = other.label
    if (this.data.length < this._width * this._height) {
      this.data = new Array[Float](this._width * this._height)
    }

    var i = 0
    while (i < this._width * this._height) {
      this.data(i) = other.data(i)
      i += 1
    }
    this
  }
}

class RGBImage(d: Array[Float], w: Int, h: Int, l: Float) extends Image(d, w, h, l) {
  def this() = this(new Array[Float](0), 0, 0, 0)

  def this(_width: Int, _height: Int) =
    this(new Array[Float](_width * _height * 3), _width, _height, 0.0f)

  def copy(rawData: Array[Byte], scale: Float = 255.0f): this.type = {
    val buffer = ByteBuffer.wrap(rawData)
    _width = buffer.getInt
    _height = buffer.getInt
    require(rawData.length == 8 + _width * _height * 3)
    if (data.length < _height * _width * 3) {
      data = new Array[Float](_width * _height * 3)
    }
    var i = 0
    while (i < _width * _height * 3) {
      data(i) = (rawData(i + 8) & 0xff) / scale
      i += 1
    }
    this
  }

  def save(path: String, scale: Float = 255.0f): Unit = {
    val image = new BufferedImage(width(), height(), BufferedImage.TYPE_INT_BGR)
    var y = 0
    while (y < height()) {
      var x = 0
      while (x < width()) {
        val r = (data((x + y * width()) * 3 + 2) * scale).toInt
        val g = (data((x + y * width()) * 3 + 1) * scale).toInt
        val b = (data((x + y * width()) * 3) * scale).toInt
        image.setRGB(x, y, (r << 16) | (g << 8) | b)
        x += 1
      }
      y += 1
    }

    ImageIO.write(image, "jpg", new File(path))
  }

  def copy(other: RGBImage): RGBImage = {
    this._width = other._width
    this._height = other._height
    this._label = other._label
    if (this.data.length < this._width * this._height * 3) {
      this.data = new Array[Float](this._width * this._height * 3)
    }

    var i = 0
    while (i < this._width * this._height * 3) {
      this.data(i) = other.data(i)
      i += 1
    }
    this
  }
}

object RGBImage {
  def readImage(path: Path, scaleTo: Int): Option[Array[Byte]] = {
    var fis : FileInputStream = null
    try {
      fis = new FileInputStream(path.toString)
      val channel = fis.getChannel
      val byteArrayOutputStream = new ByteArrayOutputStream
      channel.transferTo(0, channel.size, Channels.newChannel(byteArrayOutputStream))
      val img = ImageIO.read(new ByteArrayInputStream(byteArrayOutputStream.toByteArray))
      if (img.getAlphaRaster != null) {
        throw new UnsupportedOperationException("Not support img with alpha channel")
      }

      val heightAfterScale = if (img.getWidth < img.getHeight) {
        scaleTo * img.getHeight / img.getWidth
      } else {
        scaleTo
      }
      val widthAfterScale = if (img.getWidth < img.getHeight) {
        scaleTo
      } else {
        scaleTo * img.getWidth / img.getHeight
      }

      val scaledImage: java.awt.Image =
        img.getScaledInstance(widthAfterScale, heightAfterScale, java.awt.Image.SCALE_SMOOTH)
      val imageBuff: BufferedImage =
        new BufferedImage(widthAfterScale, heightAfterScale, BufferedImage.TYPE_3BYTE_BGR)
      imageBuff.getGraphics.drawImage(scaledImage, 0, 0, new Color(0, 0, 0), null)
      val pixels: Array[Byte] =
        (imageBuff.getRaster.getDataBuffer.asInstanceOf[DataBufferByte]).getData
      require(pixels.length % 3 == 0)

      val bytes = new Array[Byte](8 + pixels.length)
      val byteBuffer = ByteBuffer.wrap(bytes)
      byteBuffer.putInt(imageBuff.getWidth)
      byteBuffer.putInt(imageBuff.getHeight)
      System.arraycopy(pixels, 0, bytes, 8, pixels.length)
      Some(bytes)
    } catch {
      case ex: Exception =>
        ex.printStackTrace
        System.err.println("Can't read file " + path)
        None
    } finally {
      if(fis != null) {
        fis.close()
      }
    }
  }
}

abstract class ArrayDataSource[T, D](looped: Boolean) extends DataSource[D] {
  private var offset = 0

  protected val data: Array[T]

  override def shuffle(): Unit = {
    var i = 0
    while (i < data.length) {
      val exchange = i + RandomGenerator.RNG.uniform(0, data.length - i).toInt
      val tmp = data(exchange)
      data(exchange) = data(i)
      data(i) = tmp
      i += 1
    }
  }

  override def reset(): Unit = {
    offset = 0
  }

  override def next(): D = {
    val r = convert(data(if (looped) (offset % data.length) else offset))
    offset += 1
    r
  }

  def convert(rawData: T): D

  override def finished(): Boolean = (offset >= data.length)

  override def hasNext: Boolean = {
    if (looped) {
      true
    } else {
      offset < data.length
    }
  }

  override def total(): Long = data.length
}

class MNISTDataSource(trainDataPath: String, validationDataPath: String, looped: Boolean)
  extends ArrayDataSource[Array[Byte], GreyImage](looped) {
  private val ROW_N = 28
  private val COL_N = 28

  private val buffer = new GreyImage(ROW_N, COL_N)

  override val data = load(trainDataPath, validationDataPath)

  private def load(featureFile: String, labelFile: String): Array[Array[Byte]] = {
    val labelBuffer = ByteBuffer.wrap(Files.readAllBytes(Paths.get(labelFile)))
    val featureBuffer = ByteBuffer.wrap(Files.readAllBytes(Paths.get(featureFile)))
    val labelMagicNumber = labelBuffer.getInt()

    require(labelMagicNumber == 2049)
    val featureMagicNumber = featureBuffer.getInt()
    require(featureMagicNumber == 2051)

    val labelCount = labelBuffer.getInt()
    val featureCount = featureBuffer.getInt()
    require(labelCount == featureCount)

    val rowNum = featureBuffer.getInt()
    require(rowNum == ROW_N)
    val colNum = featureBuffer.getInt()
    require(colNum == COL_N)

    val result = new Array[Array[Byte]](featureCount)
    var i = 0
    while (i < featureCount) {
      val img = new Array[Byte]((rowNum * colNum + 1))
      img(0) = labelBuffer.get()
      var y = 0
      while (y < rowNum) {
        var x = 0
        while (x < colNum) {
          img(1 + x + y * colNum) = featureBuffer.get()
          x += 1
        }
        y += 1
      }
      result(i) = img
      i += 1
    }

    result
  }

  override def convert(rawData: Array[Byte]): GreyImage = {
    buffer.setLabel(rawData(0).toFloat + 1).copy(rawData, 255.0f, 1)
  }
}

class CifarDataSource(path: Path, looped: Boolean, scaleTo: Int = 32)
  extends ArrayDataSource[(Float, Array[Byte]), RGBImage](looped) with DirectoryAsLabelDataSet {
  private val buffer = new RGBImage()

  private val paths = loadPaths(path)

  override protected val data: Array[(Float, Array[Byte])] = paths.map(imageFile => {
    RGBImage.readImage(imageFile._2, scaleTo) match {
      case Some(img) => Some(imageFile._1.toFloat, img)
      case None => None
    }
  }).filter(_.isDefined).map(_.get)

  override def convert(rawData: (Float, Array[Byte])): RGBImage = {
    buffer.copy(rawData._2).setLabel(rawData._1)
  }
}

class ImageNetDataSource(path: Path, looped: Boolean, scaleTo: Int = 256)
  extends ArrayDataSource[(Float, Path), RGBImage](looped) with DirectoryAsLabelDataSet {

  override val data: Array[(Float, Path)] = loadPaths(path)

  private val buffer = new RGBImage()

  override def convert(rawData: (Float, Path)): RGBImage = {
    val imgData = RGBImage.readImage(rawData._2, scaleTo)
    val label = rawData._1
    if (imgData.isDefined) {
      buffer.copy(imgData.get).setLabel(label)
    } else {
      null
    }
  }
}

trait DirectoryAsLabelDataSet {
  def loadPaths(path: Path): Array[(Float, Path)] = {
    Class.forName("javax.imageio.ImageIO")
    Class.forName("java.awt.color.ICC_ColorSpace")
    Class.forName("sun.java2d.cmm.lcms.LCMS")
    ColorSpace.getInstance(ColorSpace.CS_sRGB).toRGB(Array[Float](0, 0, 0))

    val directoryStream = Files.newDirectoryStream(path)
    println(s"Start to read directories $path")
    val labelMap = getLabelMap(path)
    import scala.collection.JavaConverters._
    directoryStream.asScala.flatMap(dir => {
      println(s"Find class ${dir.getFileName} -> ${labelMap(dir.getFileName.toString)}")
      Files.newDirectoryStream(dir).asScala.map(p =>
        (labelMap(dir.getFileName.toString).toFloat, p)).toSeq
    }).toArray.sortWith(
      _._2.getFileName.toString < _._2.getFileName.toString
    )
  }

  def getLabelMap(path: Path): Map[String, Int] = {
    import scala.collection.JavaConverters._
    Files.newDirectoryStream(path).asScala.map(_.getFileName.toString)
      .toArray.sortWith(_ < _).zipWithIndex.map(c => c._1 -> (c._2 + 1)).toMap
  }
}
