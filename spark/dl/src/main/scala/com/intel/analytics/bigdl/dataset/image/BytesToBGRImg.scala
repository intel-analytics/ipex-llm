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
import java.nio.ByteBuffer

import com.intel.analytics.bigdl.dataset.{ByteRecord, Transformer}

import scala.collection.Iterator

object BytesToBGRImg {
  def apply(normalize: Float = 255f, resizeW : Int = -1, resizeH : Int = -1): BytesToBGRImg =
    new BytesToBGRImg(normalize, resizeW, resizeH)
}

/**
 * Convert a byte record to BGR image. The format is, first 4 bytes is width, the next 4 bytes is
 * height, and the last is pixels coming with BGR order.
 * @param normalize
 */
class BytesToBGRImg(normalize: Float, resizeW : Int = -1, resizeH : Int = -1)
  extends Transformer[ByteRecord, LabeledBGRImage] {

  private val buffer = new LabeledBGRImage()

  override def apply(prev: Iterator[ByteRecord]): Iterator[LabeledBGRImage] = {
    prev.map(rawData => {
      buffer.copy(getImgData(rawData, resizeW, resizeH), normalize).setLabel(rawData.label)
    })
  }

  private def getImgData (record : ByteRecord, resizeW : Int, resizeH : Int)
  : Array[Byte] = {
    if (resizeW == -1) {
      return record.data
    } else {
      val rawData = record.data
      val imgBuffer = ByteBuffer.wrap(rawData)
      val width = imgBuffer.getInt
      val height = imgBuffer.getInt
      val bufferedImage : BufferedImage
      = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR)
      val outputImagePixelData = bufferedImage.getRaster.getDataBuffer
        .asInstanceOf[DataBufferByte].getData
      System.arraycopy(imgBuffer.array(), 8,
        outputImagePixelData, 0, outputImagePixelData.length)
      BGRImage.resizeImage(bufferedImage, resizeW, resizeH)
    }
  }
}
