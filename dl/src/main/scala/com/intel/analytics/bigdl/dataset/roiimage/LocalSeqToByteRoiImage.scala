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

import com.intel.analytics.bigdl.dataset.{LocalSeqFilePath, Transformer}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.SequenceFile.Reader
import org.apache.hadoop.io.{SequenceFile, Text}

import scala.collection.Iterator

/**
 * transform local sequence file to ByteRoiImage
 */
class LocalSeqToByteRoiImage extends Transformer[LocalSeqFilePath, RoiByteImage] {

  import org.apache.hadoop.fs.{Path => hPath}


  @transient
  private var key: Text = null

  @transient
  private var value: Text = null

  @transient
  private var reader: SequenceFile.Reader = null

  @transient
  private var oneRecordBuffer: RoiByteImage = null

  @transient
  private var recordTransformer: RecordToByteImageWithRoi = null

  override def apply(prev: Iterator[LocalSeqFilePath]): Iterator[RoiByteImage] = {
    if (recordTransformer == null) recordTransformer = new RecordToByteImageWithRoi
    new Iterator[RoiByteImage] {
      override def next(): RoiByteImage = {
        if (oneRecordBuffer != null) {
          val res = oneRecordBuffer
          oneRecordBuffer = null
          return res
        }

        if (key == null) {
          key = new Text()
        }
        if (value == null) {
          value = new Text
        }
        if (reader == null || !reader.next(key, value)) {
          if (reader != null) {
            reader.close()
          }

          reader = new SequenceFile.Reader(new Configuration,
            Reader.file(new hPath(prev.next().path.toAbsolutePath.toString)))
          reader.next(key, value)
        }

        recordTransformer.transform(key, value)
      }

      override def hasNext: Boolean = {
        if (oneRecordBuffer != null) {
          true
        } else if (reader == null) {
          prev.hasNext
        } else {
          if (reader.next(key, value)) {
            oneRecordBuffer = recordTransformer.transform(key, value)
            return true
          } else {
            prev.hasNext
          }
        }
      }
    }
  }
}

object LocalSeqToByteRoiImage {
  def apply(): LocalSeqToByteRoiImage = new LocalSeqToByteRoiImage()
}

