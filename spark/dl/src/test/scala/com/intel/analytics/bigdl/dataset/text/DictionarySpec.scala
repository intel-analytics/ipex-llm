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
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.utils.SparkContextLifeCycle
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.io.Source

class DictionarySpec extends SparkContextLifeCycle with Matchers {
  override def nodeNumber: Int = 1
  override def coreNumber: Int = 1
  override def appName: String = "DictionarySpec"

  "DictionarySpec" should "creates dictionary correctly on Spark" in {
    val tmpFile = java.io.File
      .createTempFile("UnitTest", "DictionarySpec").getPath

    val sentence1 = "Enter Barnardo and Francisco, two sentinels."
    val sentence2 = "Who’s there?"
    val sentence3 = "I think I hear them. Stand ho! Who is there?"

    val sentences = Array(sentence1, sentence2, sentence3)

    new PrintWriter(tmpFile, "UTF-8") {
      write(sentences.mkString("\n")); close
    }

    val tokens = DataSet.rdd(sc.textFile(tmpFile)
      .filter(!_.isEmpty)).transform(SentenceTokenizer())
    val output = tokens.toDistributed().data(train = false)

    val numOfWords = 21

    val dictionary = Dictionary(output, 100)

    dictionary.getVocabSize() should be (numOfWords)
    dictionary.getDiscardSize() should be (0)
    dictionary.print()
    dictionary.printDiscard()
    dictionary.getVocabSize() should be (numOfWords)
    sc.stop()
  }

  "DictionarySpec" should "creates dictionary correctly on local" in {
    val tmpFile = java.io.File
      .createTempFile("UnitTest", "DictionarySpec").getPath

    val sentence1 = "Enter Barnardo and Francisco, two sentinels."
    val sentence2 = "Who’s there?"
    val sentence3 = "I think I hear them. Stand ho! Who is there?"

    val sentences = Array(sentence1, sentence2, sentence3)

    new PrintWriter(tmpFile, "UTF-8") {
      write(sentences.mkString("\n")); close
    }

    val logData = Source.fromFile(tmpFile, "UTF-8").getLines().toArray
    val tokens = DataSet.array(logData
      .filter(!_.isEmpty)).transform(SentenceTokenizer())
    val output = tokens.toLocal().data(train = false)

    val numOfWords = 21

    val dictionary = Dictionary(output, 100)

    dictionary.getVocabSize() should be (numOfWords)
    dictionary.getDiscardSize() should be (0)
    dictionary.print()
    dictionary.printDiscard()
    dictionary.getVocabSize() should be (numOfWords)
  }
}
