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

import com.intel.analytics.bigdl.dataset.Transformer

import scala.collection.Iterator
import scala.collection.mutable.ArrayBuffer

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
//import org.scalatest.FlatSpec
import org.apache.spark.ml.feature.{Tokenizer, RegexTokenizer}
import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import java.io._

import smile.nlp.tokenizer.{SimpleSentenceSplitter, SimpleTokenizer}

  /**
  * Transformer that tokenizes a Document (article)
  * into a Seq[Seq[String]] by using Stanford Tokenizer.
  *
  */

class DocumentTokenizer() extends Transformer[String, Array[Array[String]]] {
  //import edu.stanford.nlp.process.DocumentPreprocessor
  //var dp: DocumentPreprocessor = null
  override def apply(prev: Iterator[String]): Iterator[Array[Array[String]]] =
    prev.map(x => {
      //dp = new DocumentPreprocessor(x)
      val sentences = ArrayBuffer[Array[String]]()
      val sc = new SparkContext("local[1]", "DocumentTokenizer")
      //for (i <- filelist.indices){
      val logData = sc.textFile(x, 2).filter(!_.isEmpty()).cache()

      val sqlContext = new SQLContext(sc)
      import sqlContext.implicits._

      //println(logData.collect())
      val sentences_split = SimpleSentenceSplitter.getInstance.split(logData.collect().reduce((l,r)=>l+r))
      val tokenizer = new SimpleTokenizer(true)
      //val words = sentences.flatMap(tokenizer.split(_))
      for (i <- sentences_split.indices){
          val words = tokenizer.split(sentences_split(i))
          sentences.append(words)
      }
      sentences.toArray
/*      val new_df = logData.flatMap(_.split("((?<=[.!?])|(?=[.!?]))\\s")).toDF("sentence")
      val regexTokenizer = new RegexTokenizer().setInputCol("sentence").setOutputCol("words").setPattern("\\s|(?=[\\.,:?!](\\W|$))|(?<=\\W[\\.:?!])")
      val regexTokenized = regexTokenizer.transform(new_df)
      val countTokens = udf { (words: Seq[String]) => words.length }
      regexTokenized.select("sentence", "words").withColumn("tokens", countTokens(col("words"))).show(false)
      val token_df = regexTokenized.select("sentence", "words").withColumn("tokens", countTokens(col("words")))
      val mapped_vector = token_df.select("words").rdd.map(x => x(0).asInstanceOf[Seq[String]]).collect()

      for (i <- mapped_vector.indices){
          val mapped_token = mapped_vector(i).toArray
          sentences.append(mapped_token)
      }

      sentences.toArray*/
    })
}

object DocumentTokenizer {
  def apply(): DocumentTokenizer = new DocumentTokenizer()
}
