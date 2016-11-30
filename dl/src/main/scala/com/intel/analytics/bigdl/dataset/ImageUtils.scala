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
  /**
   * This method compress or expend the image to a specific size
   *
   * @param src the source image
   * @param width the width of an image to be converted
   * @param height the height of an image to be converted
   * @return the rescaled image
   */
  def scale(src: Image, width: Int, height: Int): Image = {
    val dstWidth = width.max(1)
    val dstHeight = height.max(1)
    val dst = new Image(
      new Array[Float](dstHeight * dstWidth * src.numChannels),
      ImageMetadata(dstWidth, dstHeight, src.numChannels))

    val srcHeight = src.height

    val channelNum = src.metadata.numChannels
    val tmp = new Image(
      new Array[Float](src.height * dstWidth * src.metadata.numChannels),
      ImageMetadata(dstWidth, src.height, src.numChannels))

    (0 until channelNum).foreach(k => {
      // Compress/expend rows first
      (0 until srcHeight).foreach(j => {
        linearScale(src, tmp, 2, j, k)
      })

      // then Columns
      (0 until dstWidth).foreach(i => {
        linearScale(tmp, dst, 1, i, k)
      })
    })

    dst
  }

  /**
   * Linear Scale the image on the selected channel along the selected dimension
   *
   * @param src the source image
   * @param dst the destination image
   * @param dim If dim == 1, then scale height. if dim == 2, then scale width
   * @param idx the index of the selected dimension
   * @param cdx the index of the selected channel
   */
  def linearScale(
    src: Image,
    dst: Image,
    dim: Int,
    idx: Int,
    cdx: Int): Unit = {
    var srcLength = 0
    var dstLength = 0

    if (dim == 1) {
      srcLength = src.height
      dstLength = dst.height
    } else if (dim == 2) {
      srcLength = src.width
      dstLength = dst.width
    }

    require(dim == 1 || dim == 2, "dim must be 1 or 2")

    def put(src: Image, i: Int, j: Int, channelIdx: Int, newVal: Float): Unit = {
      if (dim == 1) {
        src.put(j, i, channelIdx, newVal)
      } else {
        src.put(i, j, channelIdx, newVal)
      }
    }

    def get(src: Image, i: Int, j: Int, channelIdx: Int): Float = {
      if (dim == 1) {
        src.get(j, i, channelIdx)
      } else {
        src.get(i, j, channelIdx)
      }
    }

    if (dstLength > srcLength) {
      val scale = (srcLength - 1).toFloat / (dstLength - 1)

      if (srcLength == 1) for (di <- 0 until dstLength - 1) {
        put(dst, idx, di, cdx, get(src, idx, 0, cdx))
      } else for (di <- 0 until dstLength - 1) {
        var si_f = di * scale
        val si_i = si_f.toInt
        si_f -= si_i

        put(dst, idx, di, cdx, fromIntermediate(
          (1 - si_f) * get(src, idx, si_i, cdx)) +
          si_f * get(src, idx, si_i + 1, cdx)
        )
      }

      put(dst, idx, dstLength - 1, cdx, get(src, idx, srcLength - 1, cdx))
    } else if (dstLength < srcLength) {
      val scale = srcLength.toFloat / dstLength
      var si0_f = 0f
      var si0_i = 0

      for (di <- 0 until dstLength) {
        var si1_f = (di + 1) * scale
        val si1_i = si1_f.toInt
        si1_f -= si1_i
        var acc = (1 - si0_f) * get(src, idx, si0_i, cdx)
        var n = 1 - si0_f

        for (si <- (si0_i + 1) until si1_i) {
          acc += get(src, idx, si, cdx)
          n += 1
        }
        if (si1_i < srcLength) {
          acc += si1_f * get(src, idx, si1_i, cdx)
          n += si1_f
        }

        put(dst, idx, di, cdx, fromIntermediate(acc / n))
        si0_i = si1_i
        si0_f = si1_f
      }
    } else {
      for (di <- 0 until dstLength)
        put(dst, idx, di, cdx, get(src, idx, di, cdx))
    }

    def fromIntermediate(x: Float): Float = {
      val result: Float = x + 0.5f
      if (result <= 0) return 0
      if (result >= 255) return 255

      x
    }
  }

//  private def linearScaleRow(src: Image, dst: Image, row: Int, cdx: Int): Unit = {
//    val srcWidth = src.width
//    val dstWidth = dst.width
//
//    if (dstWidth > srcWidth) {
//      val scale = (srcWidth - 1).toFloat / (dstWidth - 1)
//
//      if (srcWidth == 1) for (di <- 0 until dstWidth - 1) {
//        dst.put(row, di, cdx, src.get(row, 0, cdx))
//      } else for (di <- 0 until dstWidth - 1) {
//        var si_f = di * scale
//        val si_i = si_f.toInt
//        si_f -= si_i
//
//        dst.put(row, di, cdx, fromIntermediate(
//          (1 - si_f) * src.get(row, si_i, cdx)) +
//          si_f * src.get(row, si_i + 1, cdx)
//        )
//      }
//
//      dst.put(row, dstWidth - 1, cdx, src.get(row, srcWidth - 1, cdx))
//    } else if (dstWidth < srcWidth) {
//      val scale = srcWidth.toFloat / dstWidth
//      var si0_f = 0f
//      var si0_i = 0
//
//      for (di <- 0 until dstWidth) {
//        var si1_f = (di + 1) * scale
//        val si1_i = si1_f.toInt
//        si1_f -= si1_i
//        var acc = (1 - si0_f) * src.get(row, si0_i, cdx)
//        var n = 1 - si0_f
//
//        for (si <- (si0_i + 1) until si1_i) {
//          acc += src.get(row, si, cdx)
//          n += 1
//        }
//        if (si1_i < srcWidth) {
//          acc += si1_f * src.get(row, si1_i, cdx)
//          n += si1_f
//        }
//
//        dst.put(row, di, cdx, fromIntermediate(acc / n))
//        si0_i = si1_i
//        si0_f = si1_f
//      }
//    } else {
//      for (di <- 0 until dstWidth)
//        dst.put(row, di, cdx, src.get(row, di, cdx))
//    }
//  }
//
//
//  private def linearScaleCol(src: Image, dst: Image, col: Int, cdx: Int): Unit = {
//    val srcHeight = src.height
//    val dstHeight = dst.height
//
//    if (dstHeight > srcHeight) {
//      val scale = (srcHeight - 1).toFloat / (dstHeight - 1)
//
//      if (srcHeight == 1) for (di <- 0 until dstHeight - 1) {
//        dst.put(di, col, cdx, src.get(0, col, cdx))
//      } else for (di <- 0 until dstHeight - 1) {
//        var si_f = di * scale
//        val si_i = si_f.toInt
//        si_f -= si_i
//
//        dst.put(di, col, cdx, fromIntermediate(
//          (1 - si_f) * src.get(si_i, col, cdx)) +
//          si_f * src.get(si_i + 1, col, cdx)
//        )
//      }
//
//      dst.put(dstHeight - 1, col, cdx, src.get(srcHeight - 1, col, cdx))
//    } else if (dstHeight < srcHeight) {
//      val scale = srcHeight.toFloat / dstHeight
//      var si0_f = 0f
//      var si0_i = 0
//
//      for (di <- 0 until dstHeight) {
//        var si1_f = (di + 1) * scale
//        val si1_i = si1_f.toInt
//        si1_f -= si1_i
//
//        var acc = (1 - si0_f) * src.get(si0_i, col, cdx)
//        var n = 1 - si0_f
//        for (si <- (si0_i + 1) until si1_i) {
//          acc += src.get(si, col, cdx)
//          n += 1
//        }
//        if (si1_i < srcHeight) {
//          acc += si1_f * src.get(si1_i, col, cdx)
//          n += si1_f
//        }
//
//        dst.put(di, col, cdx, fromIntermediate(acc / n))
//        si0_i = si1_i
//        si0_f = si1_f
//      }
//    } else {
//      for (di <- 0 until dstHeight)
//        dst.put(di, col, cdx, src.get(di, col, cdx))
//    }
//  }

  def hFlip(image: Image): Image = {
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

  final val NO_SCALE = -1

  def readBGRImage(path: Path, scaleTo: Int = 1): Array[Byte] = {
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
      if (NO_SCALE == scaleTo) {
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
