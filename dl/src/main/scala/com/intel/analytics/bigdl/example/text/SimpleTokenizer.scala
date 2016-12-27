/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
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

package com.intel.analytics.bigdl.example.text.classification

import java.util.{Set => JSet}

import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable.{ArrayBuffer, Map}

object SimpleTokenizer {

  def textToTokens(text: String, lower: Boolean = true): Array[String] = {
    text.replaceAll("[^a-zA-Z ]", " ").toLowerCase().split("\\s+").filter(_.size > 2)
  }

  def main(args: Array[String]): Unit = {
    SimpleTokenizer.textToTokens("aee,b.c/e$").foreach(println(_))
    SimpleTokenizer.textToTokens("religion.").foreach(println(_))
  }
}

class SimpleTokenizer(nbWords: Int, lower: Boolean = true, split: Char = ' ') {
  val log: Logger = LoggerFactory.getLogger(this.getClass)
  val wordToCount = Map[String, Int]().withDefaultValue(0)
  val wordToIndex = Map[String, Int]().withDefaultValue(0)
  val indexToWord = Map[Int, String]().withDefaultValue("")
  private var documentCount = 0

  /**
   * The Tokenizer should be feed with some texts before working.
   *
   * @param texts a collection of text to init the tokenizer
   */
  def init(texts: Seq[String]): Unit = {
    var t = 0
    while (t < texts.length) {
      this.documentCount += 1
      val seq: Seq[String] = SimpleTokenizer.textToTokens(texts(t), lower)
      var s = 0
      while (s < seq.length) {
        wordToCount(seq(s)) += 1
        s += 1
      }
      t += 1
    }

    val wordCounts: Array[(String, Int)] = wordToCount.toArray.sortBy[Int](-_._2)

    if (log.isDebugEnabled()) {
      wordCounts.slice(0, 10).foreach { item =>
        log.debug(s"top frequency - ${item._1}: ${item._2}")
      }
    }

    var i = 0
    // construct word_index and start from the most frequency item with 1-base indexed.
    while (i < wordCounts.length) {
      wordToIndex(wordCounts(i)._1) = i + 1
      indexToWord(i + 1) = wordCounts(i)._1
      i += 1
    }
    log.info(s"Found ${wordToIndex.size} uniqe tokens.")
  }

  /**
   * Transform each text in texts to a sequence of indexes.
   * The un-know word would be ignore
   *
   * @param texts  a collection of text
   * @param seqLen the length of the returning sequence,
   * it be would padded or truncated in case not meet the requirement.
   * @return a collection of sequences
   */
  def textsToSequences(texts: ArrayBuffer[String], seqLen: Int, trunc: String = "pre")
  : ArrayBuffer[ArrayBuffer[Int]] = {
    texts.map { text =>
      val seq: Array[String] = SimpleTokenizer.textToTokens(text, lower)
      var result = new ArrayBuffer[Int]()
      seq.foreach { word =>
        val i = wordToIndex(word)
        // put nothing in 0 position
        if (i != 0 && i <= nbWords) {
          result.append(i)
        }
      }
      // truncated or padding here
      val diffLen = seqLen - result.length
      if (diffLen < 0) {
        if ("pre" == trunc) {
          result.slice(result.length - seqLen, result.length)
        } else {
          result.slice(0, seqLen)
        }
      } else {
        result ++ Array.fill[Int](diffLen)(0)
      }
    }
  }
}
