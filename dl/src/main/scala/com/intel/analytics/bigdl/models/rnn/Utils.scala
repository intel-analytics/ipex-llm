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

package com.intel.analytics.bigdl.models.rnn

import java.io._

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.LogSoftMax
import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.spark.sql.SQLContext
import scopt.OptionParser
import org.apache.spark.SparkContext
import org.apache.spark.sql.functions._
import java.io._

import org.apache.spark.ml.feature.RegexTokenizer

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object Utils {
  case class TrainParams(
    folder: String =
      "/home/ywan/Documents/data/shakespeare/spark",
    cache: Option[String] = None,
    modelSnapshot: Option[String] = None,
    stateSnapshot: Option[String] = None,
    learningRate: Double = 0.1,
    hiddenSize: Int = 40,
    maxEpoch: Int = 10,
    coreNumber: Int = (Runtime.getRuntime().availableProcessors() / 2))

  val trainParser = new OptionParser[TrainParams]("BigDL SimpleRNN Train Example") {
    opt[String]('f', "folder")
      .text("where you put the MNIST data")
      .action((x, c) => c.copy(folder = x))

    opt[String]("model")
      .text("model snapshot location")
      .action((x, c) => c.copy(modelSnapshot = Some(x)))
    opt[String]("state")
      .text("state snapshot location")
      .action((x, c) => c.copy(stateSnapshot = Some(x)))

    opt[String]("cache")
      .text("where to cache the model")
      .action((x, c) => c.copy(cache = Some(x)))

    opt[Double]('r', "learningRate")
      .text("learning rate")
      .action((x, c) => c.copy(learningRate = x))

    opt[Int]('h', "hidden")
      .text("hidden size")
      .action((x, c) => c.copy(hiddenSize = x))

    opt[Int]('e', "maxEpoch")
      .text("epoch numbers")
      .action((x, c) => c.copy(maxEpoch = x))

    opt[Int]('c', "core")
      .text("cores number to train the model")
      .action((x, c) => c.copy(coreNumber = x))
  }

  class WordTokenizer(
    inputFile: String,
    saveDirectory: String,
    dictionaryLength: Int,
    sc: SparkContext)
    extends Serializable{

    def process() {
      if (!new File(saveDirectory + "/train.txt").exists) {
        // Read Input Data
        val logData = sc.textFile(inputFile, 2).filter(!_.isEmpty()).cache()

        // Special Words
        val sentence_start_token = "SENTENCE_START"
        val sentence_end_token = "SENTENCE_END"
        val unknown_token = "UNKNOWN_TOKEN"

        // Transform Input Data to DataFrame
        val sqlContext = new SQLContext(sc)
        import sqlContext.implicits._
        val input_df = logData.toDF("sentence")

        // Append Sentence Start and End to Each Sentence
        val appendSentence = udf { (sentence: String) =>
          if (!sentence.isEmpty) sentence_start_token + " " + sentence + " " + sentence_end_token else sentence
        }
        val new_df = input_df.withColumn("sentence", appendSentence(input_df("sentence")))

        // Tokenize Sentence to Words
        val regexTokenizer = new RegexTokenizer().setInputCol("sentence")
          .setOutputCol("words").setPattern("\\W+").setMinTokenLength(0)
        val regexTokenized = regexTokenizer.transform(new_df)
        val countTokens = udf { (words: Seq[String]) => words.length }
        regexTokenized.select("sentence", "words").withColumn("tokens", countTokens(col("words"))).show(false)
        val token_df = regexTokenized.select("sentence", "words").withColumn("tokens", countTokens(col("words")))

        // Create Frequency Dictionary
        val tokens = logData.flatMap(_.split("\\W+"))
        val freq_dict = tokens.filter(_.length > 0).map(word =>
          (word, 1)).reduceByKey((a, b) => a + b).sortBy[Int]((pair: Tuple2[String, Int]) => -pair._2)

        // Selecting Most Common Words According to Vocabulary
        val vocabulary_size = dictionaryLength
        val vocab_dict = freq_dict.take(vocabulary_size - 2)
        new PrintWriter(saveDirectory + "/dictionary.txt") {
          write(vocab_dict.mkString("\n")); close
        }
        val distinct_words = vocab_dict.map(x => x._1)
        val words = distinct_words :+ sentence_start_token :+ sentence_end_token

        // Mapping Words to Indexes and Generate Dictionary
        val index_dict = words.zipWithIndex.toMap
        sc.parallelize(index_dict.toSeq).saveAsTextFile(saveDirectory + "/dictionary")

        // Generate Vectors According to Dictionary
        val word_match = udf { (words: Seq[String]) =>
          words.map((word: String) => index_dict.getOrElse(word, vocabulary_size))
        }
        val mapped_df = regexTokenized.withColumn("vectors", word_match(col("words")))
        val mapped_vector = mapped_df.select("vectors").rdd.map(x => x(0).asInstanceOf[Seq[Int]]).collect()
        new PrintWriter(saveDirectory + "/mapped_data.txt") {
          write(mapped_vector.map(_.mkString(",")).mkString("\n")); close
        }

        // Generate Training Data and Labels
        val train_vector = mapped_vector.map(x => x.take(x.size - 1))
        new PrintWriter(saveDirectory + "/train.txt") {
          write(train_vector.map(_.mkString(",")).mkString("\n")); close
        }
        val label_vector = mapped_vector.map(x => x.drop(1))
        new PrintWriter(saveDirectory + "/label.txt") {
          write(label_vector.map(_.mkString(",")).mkString("\n")); close
        }
      }
    }
  }

  private[bigdl] def readSentence(filedirect: String, dictionarySize: Int,
    sc: SparkContext)
  : (Array[(Seq[Int], Seq[Int])], Array[(Seq[Int], Seq[Int])]) = {
    val logData = sc.textFile(filedirect + "/mapped_data.txt", 2).filter(!_.isEmpty).cache
    val sQLContext = new SQLContext(sc)
    import sQLContext.implicits._
    val input_df = logData.toDF("sentence")

    val data_df = input_df.select("sentence").rdd.map(x => {
      val seq = x(0).asInstanceOf[String].split(",").toList.asInstanceOf[Seq[Int]]
      (seq.take(seq.length-1), seq.drop(1))
    }).collect()

    val length = data_df.length
    val seq = Random.shuffle((1 to length).toList)
    val seqTrain = seq.take(Math.floor(seq.length*0.8).toInt).toArray
    val seqVal   = seq.drop(Math.floor(seq.length*0.8).toInt).toArray

    val trainData = seqTrain.collect(data_df)
    val valData = seqVal.collect(data_df)

    (trainData, valData)
  }

//  private[bigdl] def predict(model: Module[Float], input: Tensor[Float]): Array[Int] = {
//    val logSoftMax = LogSoftMax[Float]()
//    val _output = model.forward(input).asInstanceOf[Tensor[Float]]
//    val output = logSoftMax.forward(_output)
//
//    val outputIndex = output.max(2)._2
//        .storage.array
//        .map(_.toInt)
//    outputIndex
//  }
}
