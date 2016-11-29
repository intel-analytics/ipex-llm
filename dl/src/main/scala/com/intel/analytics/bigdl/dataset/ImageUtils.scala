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

import java.awt.Color
import java.awt.image.{BufferedImage, DataBufferByte}
import java.io.{ByteArrayInputStream, ByteArrayOutputStream, File, FileInputStream}
import java.nio.ByteBuffer
import java.nio.channels.Channels
import java.nio.file.Path
import javax.imageio.ImageIO

import com.intel.analytics.bigdl.utils.RandomGenerator._

object ImageUtils {
  def hflip(image: Image): Image = {
    hflip(image.data, image.height, image.width, image.numChannels)
    image
  }

  def hflip(data : Array[Float], height : Int, width : Int, numChannels: Int): Unit = {
    var y = 0
    while (y < height) {
      var x = 0
      while (x < width / 2) {
        var c = 0
        while (c < numChannels) {
          var swap = 0.0f
          swap = data((y * width + x) * numChannels + c)
          data((y * width + x) * numChannels + c) =
            data((y * width + width - x - 1) * numChannels + c)
          data((y * width + width - x - 1) * numChannels + c) = swap
          c += 1
        }
        x += 1
      }
      y += 1
    }
  }

  val NO_SCALE = -1

  def readRGBImage(path: Path, scaleTo: Int): Array[Byte] = {
    var fis: FileInputStream = null
    try {
      fis = new FileInputStream(path.toString)
      val channel = fis.getChannel
      val byteArrayOutputStream = new ByteArrayOutputStream
      channel.transferTo(0, channel.size, Channels.newChannel(byteArrayOutputStream))
      val img = ImageIO.read(new ByteArrayInputStream(byteArrayOutputStream.toByteArray))
      var heightAfterScale = 0
      var widthAfterScale = 0
      var scaledImage: java.awt.Image = null
      // no scale
      if (-1 == scaleTo) {
        heightAfterScale = img.getHeight
        widthAfterScale = img.getWidth
        scaledImage = img
      } else {
        if (img.getWidth < img.getHeight) {
          heightAfterScale = scaleTo * img.getHeight / img.getWidth
          widthAfterScale = scaleTo
        } else {
          heightAfterScale = scaleTo
          widthAfterScale = scaleTo * img.getWidth / img.getHeight
        }
        scaledImage =
          img.getScaledInstance(widthAfterScale, heightAfterScale, java.awt.Image.SCALE_SMOOTH)
      }

      val imageBuff: BufferedImage =
        new BufferedImage(widthAfterScale, heightAfterScale, BufferedImage.TYPE_3BYTE_BGR)
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

  def saveRGB(img: Image, path: String, scale: Float = 255.0f): Unit = {
    require(img.numChannels == 3, "RGB Image should have three channels")
    val image = new BufferedImage(
      img.width, img.height, BufferedImage.TYPE_INT_BGR)
    var y = 0
    while (y < img.height) {
      var x = 0
      while (x < img.height) {
        val r = (img.content((x + y * img.width) * 3 + 2) * scale).toInt
        val g = (img.content((x + y * img.width) * 3 + 1) * scale).toInt
        val b = (img.content((x + y * img.width) * 3) * scale).toInt
        image.setRGB(x, y, (r << 16) | (g << 8) | b)
        x += 1
      }
      y += 1
    }

    ImageIO.write(image, "jpg", new File(path))
  }

  def convertToByte(
    image: Image,
    buffer: Array[Byte] = null,
    scaleTo: Float = 255.0f): Array[Byte] = {
    val res = if (buffer == null) {
      new Array[Byte](image.height * image.width * 3)
    } else {
      require(image.height * image.width <= buffer.length)
      buffer
    }

    var i = 0
    while (i < image.height * image.width * 3) {
      res(i) = (image.data(i) * scaleTo).toByte
      i += 1
    }
    res
  }

  def crop(img: Image, cropWidth: Int, cropHeight: Int): Image = {
    val image = new Image(0, 0, 0)
    crop(image, cropWidth, cropHeight, image)
  }

  def crop(img: Image, cropWidth: Int, cropHeight: Int, buffer: Image): Image = {
    val width = img.width
    val height = img.height
    val numChannels = img.numChannels
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
  }

  def normalize(
    img: Image,
    means: Array[Double],
    stds: Array[Double]): Image = {
    val content = img.content
    require(content.length % 3 == 0)
    var i = 0
    while (i < content.length) {
      var c = 0
      while (c < img.numChannels) {
        content(i + c) = ((content(i + c) - means(c)) / stds(c)).toFloat
        c += 1
      }
      i += img.numChannels
    }
    img
  }
}
