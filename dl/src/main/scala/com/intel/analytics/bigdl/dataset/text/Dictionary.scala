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

import java.io.{File, Serializable}

import org.apache.log4j.Logger
import scala.util.Random

 /**
  * Class that help build a dictionary
  * either from tokenized text or from saved dictionary
  *
  */

class Dictionary()
   extends Serializable {

   def vocabSize(): Int = _vocabSize

   def discardSize(): Int = _discardSize

   def word2Index(): Map[String, Int] = _word2index

   def index2Word(): Map[Int, String] = _index2word

   def vocabulary(): Array[String] = _vocabulary.toArray

   def discardVocab(): Array[String] = _disvardVocab.toArray

   def getIndex(word: String): Int = {
     _word2index.getOrElse(word, _vocabSize)
   }

   def getWord(index: Float): String = {
     getWord(index.toInt)
   }

   def getWord(index: Double): String = {
     getWord(index.toInt)
   }

   def getWord(index: Int): String = {
     _index2word.getOrElse(index,
       _disvardVocab(Random.nextInt(_discardSize)))
   }

   def print(): Unit = {
     _word2index.foreach(x =>
       logger.info(x._1 + " -> " + x._2))
   }

   def printDiscard(): Unit = {
     _disvardVocab.foreach(x =>
       logger.info(x))
   }

   def this(sentences: Array[Array[String]],
            vocabSize: Int) = {
     this()
     val freqDict = sentences
       .flatMap(x => x)
       .foldLeft(Map.empty[String, Int]) {
         (count, word) => count + (word -> (count.getOrElse(word, 0) + 1))
       }.toSeq.sortBy(_._2)

     // Select most common words
     val length = math.min(vocabSize, freqDict.length)
     _vocabulary = freqDict.drop(freqDict.length - length).map(_._1)
     _vocabSize = _vocabulary.length
     _word2index = _vocabulary.zipWithIndex.toMap
     _index2word = _word2index.map(x => (x._2, x._1))
     _disvardVocab = freqDict.take(freqDict.length - length).map(_._1)
     _discardSize = _disvardVocab.length
   }

   def this(directory: String) = {
     this()

     val dictionaryFile = new File(directory, "dictionary.txt")
     if (!dictionaryFile.exists()) {
       throw new IllegalArgumentException("dictionary file not exists!")
     }
     val discardFile = new File(directory, "discard.txt")
     if (!discardFile.exists()) {
       throw new IllegalArgumentException("discard file not exists!")
     }

     import scala.io.Source
     _word2index = Source.fromFile(dictionaryFile.getAbsolutePath)
       .getLines.map(_.stripLineEnd.split("->", -1))
       .map(fields => fields(0).stripSuffix(" ") -> fields(1).stripPrefix(" ").toInt)
       .toMap[String, Int]
     _index2word = _word2index.map(x => (x._2, x._1))
     _vocabulary = _word2index.keys.toSeq
     _vocabSize = _word2index.size
     _disvardVocab = Source.fromFile(discardFile.getAbsolutePath)
       .getLines().toSeq
     _discardSize = _disvardVocab.length
   }

   val logger = Logger.getLogger(getClass)
   private var _vocabSize: Int = 0
   private var _discardSize: Int = 0
   private var _word2index: Map[String, Int] = null
   private var _index2word: Map[Int, String] = null
   private var _vocabulary: Seq[String] = null
   private var _disvardVocab: Seq[String] = null
}

object Dictionary {
  def apply(sentences: Array[Array[String]],
            vocabSize: Int)
  : Dictionary = new Dictionary(sentences, vocabSize)
  def apply(directory: String)
  : Dictionary = new Dictionary(directory)
}
