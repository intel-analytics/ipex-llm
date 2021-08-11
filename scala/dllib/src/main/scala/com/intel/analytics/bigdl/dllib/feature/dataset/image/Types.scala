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

package com.intel.analytics.bigdl.dataset.image

import java.awt.Color
import java.awt.image.{BufferedImage, DataBufferByte}
import java.io.{ByteArrayInputStream, ByteArrayOutputStream, File, FileInputStream}
import java.nio.ByteBuffer
import java.nio.channels.Channels
import java.nio.file.Path
import javax.imageio.ImageIO

import com.intel.analytics.bigdl.dataset.{Image, LocalImagePath, Label}

/**
 * Represent a local file path of a image file with a float label
 *
 * @param label
 * @param p
 */
case class LocalLabeledImagePath(var label : Float, p : Path)
  extends LocalImagePath(p) with Label[Float] {

  override def setLabel(label: Float): this.type = {
    this.label = label
    this
  }
}

/**
 * Represent a grey image
 *
 * @param data
 * @param _width
 * @param _height
 */
class GreyImage(
  protected var data: Array[Float],
  protected var _width: Int,
  protected var _height: Int
) extends Image {

  def copy(source: Array[Byte], normalize: Float = 1.0f, offset: Int = 0): this.type = {
    require(data.length + offset <= source.length)
    var i = 0
    while (i < data.length) {
      data(i) = (source(i + offset) & 0xff) / normalize
      i += 1
    }
    this
  }

  def copy(other: GreyImage): GreyImage = {
    this._width = other._width
    this._height = other._height
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

  override def width(): Int = _width

  override def content: Array[Float] = data

  override def height(): Int = _height
}

/**
 * A grey image with a float label
 *
 * @param d
 * @param w
 * @param h
 * @param _label
 */
class LabeledGreyImage(d: Array[Float], w: Int, h: Int,
  protected var _label : Float
) extends GreyImage(d, w, h) with Label[Float] {

  def this(_width: Int, _height: Int) =
    this(new Array[Float](_width * _height), _width, _height, 0.0f)

  def this() = this(new Array[Float](0), 0, 0, 0.0f)

  override def setLabel(label: Float): LabeledGreyImage.this.type = {
    this._label = label
    this
  }

  override def label(): Float = _label

  def copy(other: LabeledGreyImage): GreyImage = {
    this.copy(other.asInstanceOf[GreyImage])
    this._label = other._label
    this
  }
}

/**
 * Represent a BGR image.
 * @param data
 * @param _width
 * @param _height
 */
class BGRImage(
  protected var data: Array[Float],
  protected var _width: Int,
  protected var _height: Int
) extends Image {

  def this() = this(new Array[Float](0), 0, 0)

  def this(_width: Int, _height: Int) =
    this(new Array[Float](_width * _height * 3), _width, _height)

  def copy(rawData: Array[Byte], normalize: Float = 255.0f): this.type = {
    val buffer = ByteBuffer.wrap(rawData)
    _width = buffer.getInt
    _height = buffer.getInt
    require(rawData.length == 8 + _width * _height * 3)
    if (data.length < _height * _width * 3) {
      data = new Array[Float](_width * _height * 3)
    }
    var i = 0
    while (i < _width * _height * 3) {
      data(i) = (rawData(i + 8) & 0xff) / normalize
      i += 1
    }
    this
  }

  def copyTo(storage: Array[Float], offset: Int, toRGB: Boolean = true): Unit = {
    val frameLength = width() * height()
    require(frameLength * 3 + offset <= storage.length)
    var j = 0
    if(toRGB) {
      while (j < frameLength) {
        storage(offset + j) = content(j * 3 + 2)
        storage(offset + j + frameLength) = content(j * 3 + 1)
        storage(offset + j + frameLength * 2) = content(j * 3)
        j += 1
      }
    } else {
      while (j < frameLength) {
        storage(offset + j) = content(j * 3)
        storage(offset + j + frameLength) = content(j * 3 + 1)
        storage(offset + j + frameLength * 2) = content(j * 3 + 2)
        j += 1
      }
    }
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

  def copy(other: BGRImage): BGRImage = {
    this._width = other._width
    this._height = other._height
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

  /**
   * horizontal flip the image, modify itself
   *
   * @return
   */
  def hflip(): BGRImage = {
    BGRImage.hflip(data, height(), width())
    this
  }

  def convertToByte(buffer: Array[Byte] = null, normalize: Float = 255.0f): Array[Byte] = {
    val res = if (buffer == null) {
      new Array[Byte](height() * width() * 3)
    } else {
      require(height() * width() <= buffer.length)
      buffer
    }

    var i = 0
    while (i < height() * width() * 3) {
      res(i) = (data(i) * normalize).toByte
      i += 1
    }
    res
  }

  override def clone(): BGRImage = {
    new BGRImage().copy(this)
  }

  override def width(): Int = _width

  override def content: Array[Float] = data

  override def height(): Int = _height
}


/**
 * A BGR Image with label.
 * @param d data
 * @param w width
 * @param h height
 * @param _label a float label
 */
class LabeledBGRImage(d: Array[Float], w: Int, h: Int,
  protected var _label : Float) extends BGRImage(d, w, h) with Label[Float] {

  def this() = this(new Array[Float](0), 0, 0, 0.0f)

  def this(_width: Int, _height: Int) =
    this(new Array[Float](_width * _height * 3), _width, _height, 0.0f)

  override def setLabel(label: Float): this.type = {
    this._label = label
    this
  }

  override def hflip(): LabeledBGRImage = {
    super.hflip()
    this
  }

  override def label(): Float = _label

  def copy(other: LabeledBGRImage): LabeledBGRImage = {
    this.copy(other.asInstanceOf[BGRImage])
    this._label = other._label
    this
  }

  override def clone(): LabeledBGRImage = {
    new LabeledBGRImage().copy(this)
  }

}

object BGRImage {
  def hflip(data : Array[Float], height : Int, width : Int): Unit = {
    var y = 0
    while (y < height) {
      var x = 0
      while (x < width / 2) {
        var swap = 0.0f
        swap = data((y * width + x) * 3)
        data((y * width + x) * 3) = data((y * width + width - x - 1) * 3)
        data((y * width + width - x - 1) * 3) = swap

        swap = data((y * width + x) * 3 + 1)
        data((y * width + x) * 3 + 1) = data((y * width + width - x - 1) * 3 + 1)
        data((y * width + width - x - 1) * 3 + 1) = swap

        swap = data((y * width + x) * 3 + 2)
        data((y * width + x) * 3 + 2) = data((y * width + width - x - 1) * 3 + 2)
        data((y * width + width - x - 1) * 3 + 2) = swap
        x += 1
      }
      y += 1
    }
  }

  private def getWidthHeightAfterRatioScale(oriHeight: Int, oriWidth: Int,
    scaleTo: Int): (Int, Int) = {
    if (NO_SCALE == scaleTo) {
      (oriHeight, oriWidth)
    } else {
      if (oriWidth < oriHeight) {
        (scaleTo * oriHeight / oriWidth, scaleTo)
      } else {
        (scaleTo, scaleTo * oriWidth / oriHeight)
      }
    }
  }

  private def readRawImage(path: Path): BufferedImage = {
    var fis: FileInputStream = null
    try {
      fis = new FileInputStream(path.toString)
      val channel = fis.getChannel
      val byteArrayOutputStream = new ByteArrayOutputStream
      channel.transferTo(0, channel.size, Channels.newChannel(byteArrayOutputStream))
      val image = ImageIO.read(new ByteArrayInputStream(byteArrayOutputStream.toByteArray))
      require(image != null, "Can't read file " + path + ", ImageIO.read is null")
      image
    } catch {
      case ex: Exception =>
        ex.printStackTrace()
        System.err.println("Can't read file " + path)
        throw ex
    } finally {
      if (fis != null) {
        fis.close()
      }
    }
  }

  val NO_SCALE = -1

  def resizeImage(img: BufferedImage, resizeWidth: Int, resizeHeight: Int): Array[Byte] = {
    var scaledImage: java.awt.Image = null
    // no scale
    if ((resizeHeight == img.getHeight) && (resizeWidth == img.getWidth)) {
      scaledImage = img
    } else {
      scaledImage =
        img.getScaledInstance(resizeWidth, resizeHeight, java.awt.Image.SCALE_SMOOTH)
    }

    val imageBuff: BufferedImage =
      new BufferedImage(resizeWidth, resizeHeight, BufferedImage.TYPE_3BYTE_BGR)
    imageBuff.getGraphics.drawImage(scaledImage, 0, 0, new Color(0, 0, 0), null)
    val pixels: Array[Byte] =
      imageBuff.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData
    require(pixels.length % 3 == 0)

    val bytes = new Array[Byte](8 + pixels.length)
    val byteBuffer = ByteBuffer.wrap(bytes)
    require(imageBuff.getWidth * imageBuff.getHeight * 3 == pixels.length)
    byteBuffer.putInt(imageBuff.getWidth)
    byteBuffer.putInt(imageBuff.getHeight)
    System.arraycopy(pixels, 0, bytes, 8, pixels.length)
    bytes
  }

  def readImage(path: Path, scaleTo: Int): Array[Byte] = {
    val img: BufferedImage = readRawImage(path)
    val (heightAfterScale, widthAfterScale) =
      getWidthHeightAfterRatioScale(img.getHeight, img.getWidth, scaleTo)
    resizeImage(img, widthAfterScale, heightAfterScale)
  }

  def readImage(path: Path, resizeWidth: Int, resizeHeight: Int): Array[Byte] = {
    val img: BufferedImage = readRawImage(path)
    resizeImage(img, resizeWidth, resizeHeight)
  }
}
