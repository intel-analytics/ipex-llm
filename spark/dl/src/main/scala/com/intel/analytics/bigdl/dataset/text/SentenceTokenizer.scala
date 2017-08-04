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

import scala.collection.Iterator
import opennlp.tools.tokenize.{SimpleTokenizer, Tokenizer, TokenizerME, TokenizerModel}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}

/**
 * Transformer that tokenizes a Document (article)
 * into a Seq[Seq[String]]
 *
 */

class SentenceTokenizer(tokenFile: Option[String] = None)
  extends Transformer[String, Array[String]] {

  var modelIn: FileInputStream = _
  var model: TokenizerModel = _

  var tokenizer: Tokenizer = _

  def this(tokenFile: URL) {
    this(Some(tokenFile.getPath))
  }

  def close(): Unit = {
    if (modelIn != null) {
      modelIn.close()
    }
  }

  override def apply(prev: Iterator[String]): Iterator[Array[String]] =
    prev.map(x => {
      if (tokenizer == null) {
        if (!tokenFile.isDefined) {
          tokenizer = SimpleTokenizer.INSTANCE
        } else {
          val src: Path = new Path(tokenFile.get)
          val fs = src.getFileSystem(new Configuration())
          val in = fs.open(src)
          model = new TokenizerModel(in)
          tokenizer = new TokenizerME(model)
        }
      }
      val words = tokenizer.tokenize(x)
      words
    })
}

object SentenceTokenizer {
  def apply(tokenFile: Option[String] = None):
    SentenceTokenizer = new SentenceTokenizer(tokenFile)
  def apply(tokenFile: URL):
    SentenceTokenizer = new SentenceTokenizer(tokenFile)
}
