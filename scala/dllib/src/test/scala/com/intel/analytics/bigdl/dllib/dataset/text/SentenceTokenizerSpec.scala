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
import com.intel.analytics.bigdl.utils.{Engine, SparkContextLifeCycle}
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{FlatSpec, Matchers}

import scala.io.Source

class SentenceTokenizerSpec extends SparkContextLifeCycle with Matchers {

  override def appName: String = "DocumentTokenizer"

  "SentenceTokenizerSpec" should "tokenizes articles correctly on Spark" in {
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
        .transform(SentenceTokenizer())
    val output = tokens.toDistributed().data(train = false).collect()

    var count = 0
    println("tokenized sentences:")
    output.foreach(x => {
      count += x.length
      println(x.mkString(" "))
    })

    val numOfSents = 6
    val numOfWords = 33

    output.length should be (numOfSents)
    count should be (numOfWords)
  }

  "SentenceTokenizerSpec" should "tokenizes articles correctly on local" in {
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

    val sentenceSplitter = SentenceSplitter()
    val sentenceTokenizer = SentenceTokenizer()
    val logData = Source.fromFile(tmpFile).getLines().toArray
    val sents = DataSet.array(logData
      .filter(!_.isEmpty)).transform(sentenceSplitter)
      .toLocal().data(train = false).flatMap(item => item.iterator)
    val tokens = DataSet.array(sents.toArray)
        .transform(sentenceTokenizer)
    val output = tokens.toLocal().data(train = false).toArray

    sentenceSplitter.close()
    sentenceTokenizer.close()

    var count_word = 0
    println("tokenized sentences:")
    output.foreach(x => {
      count_word += x.length
      println(x.mkString(" "))
    })

    val numOfSents = 6
    val numOfWords = 33
    output.length should be (numOfSents)
    count_word should be (numOfWords)
  }
}
