/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.common

import com.intel.analytics.bigdl.dataset.text.Dictionary
import com.intel.analytics.bigdl.utils.RandomGenerator
import org.apache.log4j.Logger
import org.apache.spark.rdd.RDD

import scala.collection.mutable


class ZooDictionary() extends Dictionary {
  private var _vocabSize: Int = 0
  private var _discardSize: Int = 0
  private var _word2index: mutable.Map[String, Int] = null
  private var _index2word: mutable.Map[Int, String] = null
  private var _vocabulary: Seq[String] = null
  private var _discardVocab: Seq[String] = null
  @transient
  private val logger = Logger.getLogger(getClass)
  @transient
  private val rng = RandomGenerator.RNG

  /**
   * The length of the vocabulary
   */
  override def getVocabSize(): Int = _vocabSize

  /**
   * Selected words with top-k frequencies and discarded the remaining words.
   * Return the length of the discarded words.
   */
  override def getDiscardSize(): Int = _discardSize

  /**
   * Return the array of all selected words.
   */
  override def vocabulary(): Array[String] = _vocabulary.toArray

  /**
   * Return the array of all discarded words.
   */
  override def discardVocab(): Array[String] = _discardVocab.toArray

  /**
   * return the encoding number of a word,
   * if word does not existed in the dictionary,
   * it will return the dictionary length as the default index.
   * @param word
   */
  override def getIndex(word: String): Int = {
    _word2index.getOrElse(word, _vocabSize)
  }

  /**
   * return the word with regard to the index,
   * if index is out of boundary, it will randomly
   * return a word in the discarded word list.
   * If discard word list is Empty, it will randomly
   * return a word in the existed dictionary.
   * @param index
   */
  override def getWord(index: Int): String = {
    _index2word.getOrElse(index,
      if (_discardSize > 0) _discardVocab(rng.uniform(0, _discardSize).toInt)
      else getWord(RandomGenerator.RNG.uniform(0, _vocabSize).toInt))
  }

  /**
   * print word-to-index dictionary
   */
  override def print(): Unit = {
    _word2index.foreach(x =>
      logger.info(x._1 + " -> " + x._2))
  }

  /**
   * print discard dictionary
   */
  override def printDiscard(): Unit = {
    _discardVocab.foreach(x =>
      logger.info(x))
  }


  def addWord(word: String): Unit = {
    _word2index.update(word, _vocabSize)
    _index2word.update(_vocabSize, word)
    _vocabSize += 1
  }

  def this(dataset: RDD[Array[String]], vocabSize: Int) = {
    this()
    val dictionary = Dictionary(dataset, vocabSize)
    _vocabSize = dictionary.getVocabSize()
    _word2index = mutable.Map(dictionary.word2Index().toSeq: _*)
    _index2word = mutable.Map(dictionary.index2Word().toSeq: _*)
    _vocabulary = dictionary.vocabulary().toSeq
    _discardVocab = dictionary.discardVocab()
    _discardSize = _discardVocab.size
  }

  def this(index2word: Map[Int, String],
            word2index: Map[String, Int]) = {
    this()
    _index2word = mutable.Map(index2word.toSeq: _*)
    _word2index = mutable.Map(word2index.toSeq: _*)
    _vocabulary = word2index.keySet.toSeq
    _vocabSize = _vocabulary.length
  }
}
