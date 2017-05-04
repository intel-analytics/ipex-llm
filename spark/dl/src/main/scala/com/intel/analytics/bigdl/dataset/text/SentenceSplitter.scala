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

package com.intel.analytics.bigdl.dataset.text

import java.io.FileInputStream
import java.net.{URI, URL}

import com.intel.analytics.bigdl.dataset.Transformer
import opennlp.tools.sentdetect.{SentenceDetector, SentenceDetectorME, SentenceModel}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}

import scala.collection.Iterator

/**
 * Input a sequence of string, cut it into sentences.
 * The sentenceDetector is an API from OpenNLP.
 * If sentFile is None, the default sentence delimiter is period.
 * @param sentFile A trained model by OpenNLP
 */
class SentenceSplitter(sentFile: Option[String] = None)
  extends Transformer[String, Array[String]] {

  var modelIn: FileInputStream = _
  var model: SentenceModel = _
  var sentenceDetector: SentenceDetector = _

  def this(sentFileURL: URL) {
    this(Some(sentFileURL.getPath))
  }

  def this(sentFile: String) {
    this(Some(sentFile))
  }

  def close(): Unit = {
    if (modelIn != null) {
      modelIn.close()
    }
  }

  override def apply(prev: Iterator[String]): Iterator[Array[String]] =
    prev.map(x => {
      if (!sentFile.isDefined) {
        x.split('.')
      } else {
        if (sentenceDetector == null) {
          val src: Path = new Path(sentFile.get)
          val fs = src.getFileSystem(new Configuration())
          val in = fs.open(src)

          model = new SentenceModel(in)
          sentenceDetector = new SentenceDetectorME(model)
        }
        sentenceDetector.sentDetect(x)
      }
    })
}

object SentenceSplitter {
  def apply(sentFile: Option[String] = None):
    SentenceSplitter = new SentenceSplitter(sentFile)
  def apply(sentFileURL: URL):
    SentenceSplitter = new SentenceSplitter(sentFileURL)
  def apply(sentFile: String):
  SentenceSplitter = new SentenceSplitter(sentFile)
}
