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

import com.intel.analytics.bigdl.dataset.{DataSet, Transformer}

import scala.collection.{Iterator, immutable}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.spark.sql.SparkSession

import scala.io.Source

/**
  * Transformer that vectorize a Document (article)
  * into a Seq[Seq[String]]
  *
  */

class Doc2Vec(glove6BFilePath: String) extends Transformer[String, Array[Float]] {

  @transient lazy val p = new Path(glove6BFilePath)
  @transient lazy val fs = p.getFileSystem(new Configuration())
  @transient lazy val isExist = fs.exists(p)

  require(isExist, "please download glove.6B file from http://nlp.stanford.edu/data/glove.6B.zip")

  val spark = SparkSession.builder().getOrCreate()

  val libMap = spark.sparkContext.broadcast(loadWordVecMap(glove6BFilePath))

  def loadWordVecMap(filename: String): Map[String, Array[Float]] = {

    val wordMap = for (line <- spark.sparkContext.textFile(filename).collect()) yield {
      val values = line.split(" ")
      val word = values(0)
      val coefs: Array[Float] = values.slice(1, values.length).map(_.toFloat)
      (word, coefs)
    }

    wordMap.toMap
  }

  override def apply(docs: Iterator[String]): Iterator[Array[Float]] = {

    docs.map(doc => {

      val sents = DataSet.array(doc.split("\n")
        .filter(!_.isEmpty)).transform(SentenceSplitter())
        .toLocal().data(train = false).flatMap(item => item.iterator)

      val tokens = DataSet.array(sents.toArray)
        .transform(SentenceTokenizer()).toLocal().data(train = false).toArray.flatten

      val wordCount: Map[String, Int] = tokens
        .map(x => (x, 1))
        .groupBy(x => x._1)
        .map(x => (x._1, x._2.map(y => y._2).sum))

      val n = tokens.length

      // wordcount -> matrix
      /*
      *  map(is->10, I -> 20...) -> list(Array[1,2,3,4,4,4], Array[5,6,7,8,9,0]....)
       */
      val wordFreqVec: immutable.Iterable[Array[Float]] = wordCount
        .filter(p => libMap.value.contains(p._1))
        .map(x => libMap.value(x._1).map(v => v * x._2 / n)
        )

      // matrix summarize by column to array
      val docVec: Array[Float] = wordFreqVec
        .flatMap(x => x.zipWithIndex.map(x => (x._2, x._1)))
        .groupBy(x => x._1)
        .map(x => (x._1, x._2.map(_._2).sum)).toArray
        .sortBy(x => x._1)
        .map(x => x._2)

      docVec
    })
  }
}

object Doc2Vec {

  def apply(tokenFile: String): Doc2Vec = new Doc2Vec(tokenFile)

}
