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

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.dataset.text.utils.SentenceToken

import scala.collection.Iterator

/**
 * x =>  ["start", x, "end"]
 */

class SentenceBiPadding(
  start: Option[String] = None,
  end: Option[String] = None)
  extends Transformer[String, String] {

  val sentenceStart = start.getOrElse(SentenceToken.start)
  val sentenceEnd = end.getOrElse(SentenceToken.end)

  override def apply(prev: Iterator[String]): Iterator[String] = {
    prev.map(x => {
      val sentence = sentenceStart + " " + x + " " + sentenceEnd
      sentence
    })
  }
}

object SentenceBiPadding {
  def apply(start: Option[String] = None,
            end: Option[String] = None):
  SentenceBiPadding = new SentenceBiPadding(start, end)
}
