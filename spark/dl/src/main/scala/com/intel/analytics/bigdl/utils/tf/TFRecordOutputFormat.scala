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

import org.apache.hadoop.io.BytesWritable
import org.apache.hadoop.io.NullWritable
import org.apache.hadoop.mapreduce.RecordWriter
import org.apache.hadoop.mapreduce.TaskAttemptContext
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat

class TFRecordOutputFormat extends FileOutputFormat[BytesWritable, NullWritable]{
  override def getRecordWriter(taskAttemptContext: TaskAttemptContext):
  RecordWriter[BytesWritable, NullWritable] = {
    val conf = taskAttemptContext.getConfiguration
    val file = getDefaultWorkFile(taskAttemptContext, "")
    val fs = file.getFileSystem(conf)

    val bufferSize = 4096
    val outStream = fs.create(file, true, bufferSize)

    val writer = new TFRecordWriter(outStream)

    new RecordWriter[BytesWritable, NullWritable]() {
      override def close(context: TaskAttemptContext): Unit = {
        outStream.close()
      }

      override def write(k: BytesWritable, v: NullWritable): Unit = {
        writer.write(k.getBytes, 0, k.getLength)
      }
    }
  }
}
