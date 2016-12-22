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

package com.intel.analytics.bigdl.example.loadCaffe

import java.awt.Color
import java.awt.color.ColorSpace
import java.awt.image.{BufferedImage, DataBufferByte}
import java.io.{ByteArrayInputStream, ByteArrayOutputStream, FileInputStream}
import java.nio.ByteBuffer
import java.nio.channels.Channels
import java.nio.file.Path
import javax.imageio.ImageIO

import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.File

import scala.collection.Iterator


class RGBImgCropperCenter(cropWidth: Int, cropHeight: Int)
  extends Transformer[LabeledRGBImage, LabeledRGBImage] {

  private val buffer = new LabeledRGBImage(cropWidth, cropHeight)

  override def apply(prev: Iterator[LabeledRGBImage]): Iterator[LabeledRGBImage] = {
    prev.map(img => {
      val width = img.width()
      val height = img.height()
      val startW = (width - cropWidth) / 2
      val startH = (height - cropHeight) / 2
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

class RGBImgNormalizer(means: Tensor[Float])
  extends Transformer[LabeledRGBImage, LabeledRGBImage] {

  override def apply(prev: Iterator[LabeledRGBImage]): Iterator[LabeledRGBImage] = {
    prev.map(img => {
      val content = img.content
      val meansData = means.storage().array()
      require(content.length % 3 == 0)
      require(content.length == means.nElement())
      var i = 0
      while (i < content.length) {
        content(i + 2) = content(i + 2) - meansData(i + 2)
        content(i + 1) = content(i + 1) - meansData(i + 1)
        content(i + 0) = content(i + 0) - meansData(i + 0)
        i += 3
      }
      img
    })
  }
}

class LocalImgReader(scaleTo: Int)
  extends Transformer[LabeledImageLocalPath, LabeledRGBImage] {
  Class.forName("javax.imageio.ImageIO")
  Class.forName("java.awt.color.ICC_ColorSpace")
  Class.forName("sun.java2d.cmm.lcms.LCMS")
  ColorSpace.getInstance(ColorSpace.CS_sRGB).toRGB(Array[Float](0, 0, 0))

  private val buffer = new LabeledRGBImage()

  override def apply(prev: Iterator[LabeledImageLocalPath]): Iterator[LabeledRGBImage] = {
    prev.map(data => {
      val imgData = readImage(data.path, scaleTo, scaleTo)
      val label = data.label
      buffer.copy(imgData, 1f).setLabel(label)
    })
  }

  def readImage(path: Path, scaleHTo: Int, scaledWTo: Int): Array[Byte] = {
    var fis: FileInputStream = null
    try {
      fis = new FileInputStream(path.toString)
      val channel = fis.getChannel
      val byteArrayOutputStream = new ByteArrayOutputStream
      channel.transferTo(0, channel.size, Channels.newChannel(byteArrayOutputStream))
      val img = ImageIO.read(new ByteArrayInputStream(byteArrayOutputStream.toByteArray))
      var scaledImage: java.awt.Image = null

      scaledImage =
        img.getScaledInstance(scaledWTo, scaleHTo, java.awt.Image.SCALE_SMOOTH)

      val imageBuff: BufferedImage =
        new BufferedImage(scaledWTo, scaleHTo, BufferedImage.TYPE_3BYTE_BGR)
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
}

object LocalDataSetAlexnet {
  def apply(path: Path, imageSize: Int, batchSize: Int)
  : LocalDataSet[Batch[Float]] = {
    val means = File.load[Tensor[Float]](
      getClass.getClassLoader.getResource("caffe/means.obj").getPath)
    val ds = DataSet.ImageFolder.paths(path)
    val imgReader = new LocalImgReader(scaleTo = 256)
    val normalizer = new RGBImgNormalizer(means)
    val cropper = new RGBImgCropperCenter(cropWidth = imageSize, cropHeight = imageSize)
    val imgToBatch = RGBImgToBatch(batchSize)
    ds -> imgReader -> normalizer -> cropper -> imgToBatch
  }
}

object LocalDatasetGooglenet {
  def apply(path: Path, imageSize: Int, batchSize: Int)
  : LocalDataSet[Batch[Float]] = {
    val ds = DataSet.ImageFolder.paths(path)
    val imgReader = new LocalImgReader(scaleTo = 256)
    val cropper = new RGBImgCropperCenter(cropWidth = imageSize, cropHeight = imageSize)
    val normalizer = RGBImgNormalizer(123, 117, 104, 1, 1, 1)
    val imgToBatch = RGBImgToBatch(batchSize)
    ds -> imgReader -> cropper -> normalizer -> imgToBatch
  }
}



