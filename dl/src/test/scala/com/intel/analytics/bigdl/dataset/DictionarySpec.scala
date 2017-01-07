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

package com.intel.analytics.bigdl.dataset.text

import java.io.PrintWriter

import com.intel.analytics.bigdl.dataset.LocalArrayDataSet
import org.scalatest.{FlatSpec, Matchers}

class DictionarySpec extends FlatSpec with Matchers {

  "DictionarySpec" should "creates dictionary correctly" in {
    val tmpFile = java.io.File
      .createTempFile("UnitTest", "DictionarySpec").getPath

    val sentence1 = "Enter Barnardo and Francisco, two sentinels."
    val sentence2 = "Who’s there?"
    val sentence3 = "I think I hear them. Stand ho! Who is there?"

    val sentences = Array(sentence1, sentence2, sentence3)

    new PrintWriter(tmpFile) {
      write(sentences.mkString("\n")); close
    }

    val dataSet = new LocalArrayDataSet(Array(tmpFile))
    val documentTokenizer = DocumentTokenizer()
    val writePipeLine = dataSet -> documentTokenizer
    val iter = writePipeLine.toLocal().data(train = false)

    val output = iter.next

    val numOfWords = 20

    val dictionary = Dictionary(output, 100)

    dictionary.vocabSize() should be (numOfWords)
    dictionary.discardSize() should be (0)
    dictionary.print()
    dictionary.printDiscard()
  }
}
