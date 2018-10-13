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
import org.apache.spark.SparkContext
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.io.Source

@com.intel.analytics.bigdl.tags.Serial
class TextToLabeledSentenceSpec extends SparkContextLifeCycle with Matchers {
  override def nodeNumber: Int = 1
  override def coreNumber: Int = 1
  override def appName: String = "TextToLabeledSentence"

  "TextToLabeledSentenceSpec" should "indexes sentences correctly on Spark" in {
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

    val tokens = DataSet.rdd(sc.textFile(tmpFile)
      .filter(!_.isEmpty))
      .transform(SentenceTokenizer())
    val output = tokens.toDistributed().data(train = false)
    val dictionary = Dictionary(output, 100)
    val textToLabeledSentence = TextToLabeledSentence[Float](dictionary)
    val labeledSentences = tokens.transform(textToLabeledSentence)
      .toDistributed().data(false).collect()
    labeledSentences.foreach(x => {
      println("input = " + x.data().mkString(","))
      println("target = " + x.label().mkString(","))
      var i = 1
      while (i < x.dataLength()) {
        x.getData(i) should be (x.getLabel(i - 1))
        i += 1
      }
    })
  }

  "TextToLabeledSentenceSpec" should "indexes sentences correctly on Local" in {
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
    val tokens = DataSet.array(logData
      .filter(!_.isEmpty))
      .transform(SentenceTokenizer())
    val output = tokens.toLocal().data(train = false)

    val dictionary = Dictionary(output, 100)
    val textToLabeledSentence = TextToLabeledSentence[Float](dictionary)
    val labeledSentences = tokens.transform(textToLabeledSentence)
      .toLocal().data(false)
    labeledSentences.foreach(x => {
      println("input = " + x.data().mkString(","))
      println("target = " + x.label().mkString(","))
      var i = 1
      while (i < x.dataLength()) {
        x.getData(i) should be (x.getLabel(i - 1))
        i += 1
      }
    })

  }
}
