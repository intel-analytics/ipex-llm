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

import java.io.PrintWriter

import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.dataset.text.utils.SentenceToken
import com.intel.analytics.bigdl.utils.{Engine, SparkContextLifeCycle}
import org.apache.spark.SparkContext
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.io.Source

@com.intel.analytics.bigdl.tags.Serial
class SentenceBiPaddingSpec extends SparkContextLifeCycle with Matchers {
  override def nodeNumber: Int = 1
  override def coreNumber: Int = 1
  override def appName: String = "DocumentTokenizer"

  "SentenceBiPaddingSpec" should "pads articles correctly on Spark" in {
    val tmpFile = java.io.File
      .createTempFile("UnitTest", "DocumentTokenizerSpec").getPath

    val sentence1 = "Enter Barnardo and Francisco, two sentinels."
    val sentence2 = "Who’s there?"
    val sentence3 = "I think I hear them. Stand ho! Who is there?"
    val sentence4 = "The Dr. lives in a blue-painted box."

    val sentences = Array(sentence1, sentence2, sentence3, sentence4)
    new PrintWriter(tmpFile) {
      write(sentences.mkString("\n")); close
    }

    val sents = DataSet.rdd(sc.textFile(tmpFile)
      .filter(!_.isEmpty)).transform(SentenceSplitter())
      .toDistributed().data(train = false).flatMap(item => item.iterator).collect()
      .asInstanceOf[Array[String]]
    val tokens = DataSet.rdd(sc.parallelize(sents))
      .transform(SentenceBiPadding())
    val output = tokens.toDistributed().data(train = false).collect()

    var count = 0
    println("padding sentences:")
    output.foreach(x => {
      count += x.length
      println(x)
      val words = x.split(" ")
      val startToken = words(0)
      val endToken = words(words.length - 1)
      startToken should be (SentenceToken.start)
      endToken should be (SentenceToken.end)
    })
    sc.stop()
  }

  "SentenceBiPaddingSpec" should "pads articles correctly on local" in {
    val tmpFile = java.io.File
      .createTempFile("UnitTest", "DocumentTokenizerSpec").getPath

    val sentence1 = "Enter Barnardo and Francisco, two sentinels."
    val sentence2 = "Who’s there?"
    val sentence3 = "I think I hear them. Stand ho! Who is there?"
    val sentence4 = "The Dr. lives in a blue-painted box."

    val sentences = Array(sentence1, sentence2, sentence3, sentence4)

    new PrintWriter(tmpFile) {
      write(sentences.mkString("\n")); close
    }

    val logData = Source.fromFile(tmpFile).getLines().toArray
    val sents = DataSet.array(logData
      .filter(!_.isEmpty)).transform(SentenceSplitter())
      .toLocal().data(train = false).flatMap(item => item.iterator)
    val tokens = DataSet.array(sents.toArray)
      .transform(SentenceBiPadding())
    val output = tokens.toLocal().data(train = false).toArray

    var count_word = 0
    println("padding sentences:")
    output.foreach(x => {
      count_word += x.length
      println(x)
      val words = x.split(" ")
      val startToken = words(0)
      val endToken = words(words.length - 1)
      startToken should be (SentenceToken.start)
      endToken should be (SentenceToken.end)
    })
  }
}
