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

package com.intel.analytics.bigdl.visualization.tensorboard

import java.io.{File, FileOutputStream}

import com.google.common.primitives.{Ints, Longs}
import com.intel.analytics.bigdl.utils.Crc32
import netty.Crc32c
import org.apache.hadoop.fs.{FSDataOutputStream, FileSystem, Path}
import org.tensorflow.util.Event

/**
 * A writer to write event protobuf to file by tensorboard's format.
 * @param file Support local path and HDFS path
 */
private[bigdl] class RecordWriter(file: Path, fs: FileSystem) {
  val outputStream = if (file.toString.startsWith("hdfs://")) {
    // FSDataOutputStream couldn't flush data to localFileSystem in time. So reading summaries
    // will throw exception.
    fs.create(file, true, 1024)
  } else {
    // Using FileOutputStream when write to local.
    new FileOutputStream(new File(file.toString))
  }
  val crc32 = new Crc32c()
  def write(event: Event): Unit = {
    val eventString = event.toByteArray
    val header = Longs.toByteArray(eventString.length.toLong).reverse
    outputStream.write(header)
    outputStream.write(Ints.toByteArray(Crc32.maskedCRC32(crc32, header).toInt).reverse)
    outputStream.write(eventString)
    outputStream.write(Ints.toByteArray(Crc32.maskedCRC32(crc32, eventString).toInt).reverse)
    if (outputStream.isInstanceOf[FSDataOutputStream]) {
      // Flush data to HDFS.
      outputStream.asInstanceOf[FSDataOutputStream].hflush()
    }
  }

  def close(): Unit = {
    outputStream.close()
  }
}
