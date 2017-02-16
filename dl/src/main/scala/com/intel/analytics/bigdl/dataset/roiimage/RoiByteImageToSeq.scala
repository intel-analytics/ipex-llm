/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
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

package com.intel.analytics.bigdl.dataset.roiimage

import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.file.{Path, Paths}
import javax.imageio.ImageIO

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.dataset.image.BGRImage
import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{Path => hPath}
import org.apache.hadoop.io.{SequenceFile, Text}

import scala.collection.Iterator

class RoiByteImageToSeq(blockSize: Int, baseFileName: Path) extends
  Transformer[RoiImagePath, String] {
  private val conf: Configuration = new Configuration
  private var index = 0
  private val preBuffer: ByteBuffer = ByteBuffer.allocate(4 * 2)
  private var classLen: Int = 0
  private var startInd: Int = 0

  override def apply(prev: Iterator[RoiImagePath]): Iterator[String] = {
    new Iterator[String] {
      override def hasNext: Boolean = prev.hasNext

      override def next(): String = {
        val fileName = baseFileName + s"_$index.seq"
        val path = new hPath(fileName)
        val writer = SequenceFile.createWriter(conf, SequenceFile.Writer.file(path),
          SequenceFile.Writer.keyClass(classOf[Text]),
          SequenceFile.Writer.valueClass(classOf[Text]))
        var i = 0
        while (i < blockSize && prev.hasNext) {
          val roidb = prev.next()
          val originalImage = BGRImage.readRawImage(Paths.get(roidb.imagePath))
          // convert BufferedImage to byte array
          val baos = new ByteArrayOutputStream()
          // use jpg will lost some information
          ImageIO.write(originalImage, "png", baos)
          baos.flush()
          val imageInByte = baos.toByteArray()
          baos.close()
          classLen = if (roidb.target.isDefined) roidb.target.get.classes.nElement() * 4 else 0
          preBuffer.putInt(imageInByte.length)
          preBuffer.putInt(classLen)
          val data: Array[Byte] = new Array[Byte](preBuffer.capacity + imageInByte.length +
            classLen * 5 * 4)
          startInd = 0
          System.arraycopy(preBuffer.array, 0, data, startInd, preBuffer.capacity)
          startInd += preBuffer.capacity
          System.arraycopy(imageInByte, 0, data, startInd, imageInByte.length)
          startInd += imageInByte.length
          if (roidb.target.isDefined) {
            val cls = tensorToBytes(roidb.target.get.classes)
            System.arraycopy(cls, 0, data, startInd, cls.length)
            startInd += cls.length
            val bbox = tensorToBytes(roidb.target.get.bboxes)
            System.arraycopy(bbox, 0, data, startInd, bbox.length)
          }
          preBuffer.clear
          val imageKey = roidb.imagePath.substring(roidb.imagePath.lastIndexOf("/") + 1,
            roidb.imagePath.lastIndexOf("."))
          writer.append(new Text(imageKey), new Text(data))
          i += 1
        }
        writer.close()
        index += 1
        fileName
      }
    }
  }

  def tensorToBytes(tensor: Tensor[Float]): Array[Byte] = {
    val boxBuffer = ByteBuffer.allocate(tensor.nElement() * 4)
    var i = 0
    val td = tensor.storage().array()
    while (i < td.length) {
      boxBuffer.putFloat(td(i))
      i += 1
    }
    boxBuffer.array()
  }
}

object RoiByteImageToSeq {
  def apply(blockSize: Int, baseFileName: Path): RoiByteImageToSeq =
    new RoiByteImageToSeq(blockSize, baseFileName)
}

