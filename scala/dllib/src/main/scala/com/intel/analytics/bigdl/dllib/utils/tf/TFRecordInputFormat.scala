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
package com.intel.analytics.bigdl.utils.tf

import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.{BytesWritable, NullWritable}
import org.apache.hadoop.mapreduce.{InputSplit, JobContext, RecordReader, TaskAttemptContext}
import org.apache.hadoop.mapreduce.lib.input.{FileInputFormat, FileSplit}
import org.apache.hadoop.fs.FSDataInputStream

class TFRecordInputFormat extends FileInputFormat[BytesWritable, NullWritable] {
  override def createRecordReader(inputSplit: InputSplit, context: TaskAttemptContext):
  RecordReader[BytesWritable, NullWritable] = new RecordReader[BytesWritable, NullWritable] {

    private var inputStream: FSDataInputStream = null
    private var reader: TFRecordIterator = null
    private var length: Long = 0L
    private var begin: Long = 0L
    private var current: Array[Byte] = null


    override def getCurrentKey: BytesWritable = {
      new BytesWritable(current)
    }

    override def getProgress: Float = {
      (inputStream.getPos - begin) / (length + 1e-6f)
    }

    override def nextKeyValue(): Boolean = {
      if (reader.hasNext) {
        current = reader.next()
        true
      } else {
        false
      }
    }

    override def getCurrentValue: NullWritable = {
      NullWritable.get()
    }

    override def initialize(split: InputSplit, context: TaskAttemptContext): Unit = {
      val conf = context.getConfiguration
      val fileSplit = split.asInstanceOf[FileSplit]
      length = fileSplit.getLength
      begin = fileSplit.getStart

      val file = fileSplit.getPath
      val fs = file.getFileSystem(conf)
      inputStream = fs.open(file, 4096)
      reader = new TFRecordIterator(inputStream)
    }

    override def close(): Unit = {
      inputStream.close()
    }
  }

  override protected def isSplitable(context: JobContext, filename: Path): Boolean = false
}
