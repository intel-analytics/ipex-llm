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

import java.nio.ByteBuffer
import java.nio.file.Path

import com.intel.analytics.bigdl.dataset.Transformer
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{Path => hadoopPath}
import org.apache.hadoop.io.{SequenceFile, Text}

import scala.collection.Iterator

object BGRImgToLocalSeqFile {
  def apply(blockSize: Int, baseFileName: Path, hasName: Boolean = false): BGRImgToLocalSeqFile = {
    new BGRImgToLocalSeqFile(blockSize, baseFileName, hasName)
  }
}

/**
 * Write a BGR image sequence into one or many hadoop sequence files.
 * @param blockSize how many images each sequence file contains
 * @param baseFileName file name, the real generated sequence file will have a No. suffix
 */
class BGRImgToLocalSeqFile(blockSize: Int, baseFileName: Path, hasName: Boolean = false) extends
  Transformer[(LabeledBGRImage, String), String] {
  private val conf: Configuration = new Configuration
  private var index = 0
  private val preBuffer: ByteBuffer = ByteBuffer.allocate(4 * 2)

  override def apply(prev: Iterator[(LabeledBGRImage, String)]): Iterator[String] = {
    new Iterator[String] {
      override def hasNext: Boolean = prev.hasNext

      override def next(): String = {
        val fileName = baseFileName + s"_$index.seq"
        val path = new hadoopPath(fileName)
        val writer = SequenceFile.createWriter(conf, SequenceFile.Writer.file(path),
          SequenceFile.Writer.keyClass(classOf[Text]),
          SequenceFile.Writer.valueClass(classOf[Text]))
        var i = 0
        while (i < blockSize && prev.hasNext) {
          val (image, imageName) = prev.next()

          preBuffer.putInt(image.width())
          preBuffer.putInt(image.height())
          val imageByteData = image.convertToByte()
          val data: Array[Byte] = new Array[Byte](preBuffer.capacity + imageByteData.length)
          System.arraycopy(preBuffer.array, 0, data, 0, preBuffer.capacity)
          System.arraycopy(imageByteData, 0, data, preBuffer.capacity, imageByteData.length)
          preBuffer.clear
          val imageKey = if (hasName) s"${imageName}\n${image.label().toInt}"
            else s"${image.label().toInt}"
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

