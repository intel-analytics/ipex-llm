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

package com.intel.analytics.zoo.feature.text

import java.io._
import java.util

import com.intel.analytics.bigdl.DataSet
import com.intel.analytics.bigdl.dataset.{DataSet, Sample}
import com.intel.analytics.zoo.feature.common.{Preprocessing, Relation, Relations}
import com.intel.analytics.zoo.feature.text.TruncMode.TruncMode
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.util.StringUtils
import org.apache.log4j.Logger
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.collection.mutable.{ArrayBuffer, Map => MMap}
import scala.io.Source
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.feature.FeatureSet
import com.intel.analytics.zoo.feature.pmem.{DRAM, MemoryType}
import org.apache.spark.sql.SQLContext

/**
 * TextSet wraps a set of TextFeature.
 */
abstract class TextSet {
  import TextSet.logger

  /**
   * Transform from one TextSet to another.
   */
  def transform(transformer: Preprocessing[TextFeature, TextFeature]): TextSet

  // scalastyle:off methodName
  // scalastyle:off noSpaceBeforeLeftBracket
  def -> (transformer: Preprocessing[TextFeature, TextFeature]): TextSet = {
    this.transform(transformer)
  }

  /**
   * Whether it is a LocalTextSet.
   */
  def isLocal: Boolean

  /**
   * Whether it is a DistributedTextSet.
   */
  def isDistributed: Boolean

  /**
   * Convert to a LocalTextSet.
   */
  def toLocal(): LocalTextSet

  /**
   * Convert to a DistributedTextSet.
   *
   * Need to specify SparkContext to convert a LocalTextSet to a DistributedTextSet.
   * In this case, you may also want to specify partitionNum, the default of which is 4.
   */
  def toDistributed(sc: SparkContext = null, partitionNum: Int = 4): DistributedTextSet

  /**
   * Convert TextSet to DataSet of Sample.
   */
  def toDataSet: DataSet[Sample[Float]]

  /**
   * Randomly split into array of TextSet with provided weights.
   * Only available for DistributedTextSet for now.
   *
   * @param weights Array of Double indicating the split portions.
   */
  def randomSplit(weights: Array[Double]): Array[TextSet]

  /**
   * Do tokenization on original text.
   * See [[Tokenizer]] for more details.
   */
  def tokenize(): TextSet = {
    transform(Tokenizer())
  }

  /**
   * Do normalization on tokens.
   * Need to tokenize first.
   * See [[Normalizer]] for more details.
   */
  def normalize(): TextSet = {
    transform(Normalizer())
  }

  /**
   * Map word tokens to indices.
   * Important: Take care that this method behaves a bit differently for training and inference.
   *
   * ---------------------------------------Training--------------------------------------------
   * During the training, you need to generate a new wordIndex map according to the texts you are
   * dealing with. Thus this method will first do the map generation and then convert words to
   * indices based on the generated map.
   * You can specify the following arguments which poses some constraints when generating the map.
   * In the result map, index will start from 1 and corresponds to the occurrence frequency of
   * each word sorted in descending order.
   * After word2idx, you can get the generated wordIndex map by calling 'getWordIndex'.
   * Also, you can call `saveWordIndex` to save this wordIndex map to be used in future training.
   *
   * @param removeTopN Non-negative integer. Remove the topN words with highest frequencies in
   *                   the case where those are treated as stopwords.
   *                   Default is 0, namely remove nothing.
   * @param maxWordsNum Integer. The maximum number of words to be taken into consideration.
   *                    Default is -1, namely all words will be considered. Otherwise, it should
   *                    be a positive integer.
   * @param minFreq Positive integer. Only those words with frequency >= minFreq will be taken into
   *                consideration. Default is 1, namely all words that occur will be considered.
   * @param existingMap Existing map of word index if any. Default is null and in this case a new
   *                    map with index starting from 1 will be generated.
   *                    If not null, then the generated map will preserve the word index in
   *                    existingMap and assign subsequent indices to new words.
   *
   * ---------------------------------------Inference--------------------------------------------
   * During the inference, you are supposed to use exactly the same wordIndex map in the training
   * stage instead of generating a new one.
   * Thus please be aware that you do not need to specify any of the above arguments.
   * You need to call `loadWordIndex` or `setWordIndex` beforehand for map loading.
   *
   * Need to tokenize first.
   * See [[WordIndexer]] for more details.
   */
  def word2idx(
      removeTopN: Int = 0,
      maxWordsNum: Int = -1,
      minFreq: Int = 1,
      existingMap: Map[String, Int] = null): TextSet = {
    if (wordIndex != null) {
      logger.info("Using the existing wordIndex for transformation")
    } else {
      generateWordIndexMap(removeTopN, maxWordsNum, minFreq, existingMap)
    }
    transform(WordIndexer(wordIndex))
  }

  /**
   * Shape the sequence of indices to a fixed length.
   * Need to word2idx first.
   * See [[SequenceShaper]] for more details.
   */
  def shapeSequence(
      len: Int,
      truncMode: TruncMode = TruncMode.pre,
      padElement: Int = 0): TextSet = {
    transform(SequenceShaper(len, truncMode, padElement))
  }

  /**
   * Generate BigDL Sample.
   * Need to word2idx first.
   * See [[TextFeatureToSample]] for more details.
   */
  def generateSample(): TextSet = {
    transform(TextFeatureToSample())
  }

  /**
   * Generate wordIndex map based on sorted word frequencies in descending order.
   * Return the result map, which will also be stored in 'wordIndex'.
   * Make sure you call this after tokenize. Otherwise you will get an exception.
   * See word2idx for more details.
   */
  def generateWordIndexMap(
      removeTopN: Int = 0,
      maxWordsNum: Int = 5000,
      minFreq: Int = 1,
      existingMap: Map[String, Int] = null): Map[String, Int]

  private var wordIndex: Map[String, Int] = _

  /**
   * Get the word index map of this TextSet.
   * If the TextSet hasn't been transformed from word to index, null will be returned.
   */
  def getWordIndex: Map[String, Int] = wordIndex

  /**
   * Assign a wordIndex map for this TextSet to use during word2idx.
   * If you load the wordIndex from the saved file, you are recommended to use `loadWordIndex`
   * directly.
   *
   * @param vocab Map of each word (String) and its index (integer).
   */
  def setWordIndex(vocab: Map[String, Int]): this.type = {
    wordIndex = vocab
    this
  }

  /**
   * Save wordIndex map to text file, which can be used for future inference.
   * Each separate line will be "word id".
   *
   * For LocalTextSet, save txt to a local file system.
   * For DistributedTextSet, save txt to a local or distributed file system (such as HDFS).
   *
   * @param path The path to the text file.
   */
  def saveWordIndex(path: String): Unit = {
    if (wordIndex == null) {
      throw new Exception("wordIndex is null, nothing to save. " +
        "Please transform from word to index first")
    }
  }

  /**
   * Load the wordIndex map which was saved after the training, so that this TextSet can
   * directly use this wordIndex during inference.
   * Each separate line should be "word id".
   *
   * Note that after calling `loadWordIndex`, you do not need to specify any argument when calling
   * `word2idx` in the preprocessing pipeline as now you are using exactly the loaded wordIndex for
   * transformation.
   *
   * For LocalTextSet, load txt from a local file system.
   * For DistributedTextSet, load txt from a local or distributed file system (such as HDFS).
   *
   * @param path The path to the text file.
   */
  def loadWordIndex(path: String): TextSet
}


object TextSet {

  val logger: Logger = Logger.getLogger(getClass)

  /**
   * Create a LocalTextSet from array of TextFeature.
   */
  def array(data: Array[TextFeature]): LocalTextSet = {
    new LocalTextSet(data)
  }

  /**
   * Create a DistributedTextSet from RDD of TextFeature.
   */
  def rdd(data: RDD[TextFeature], memoryType: MemoryType = DRAM): DistributedTextSet = {
    new DistributedTextSet(data, memoryType)
  }

  /**
   * Read text files with labels from a directory.
   *
   * The directory structure is expected to be the following:
   * path
   *   ├── dir1 - text1, text2, ...
   *   ├── dir2 - text1, text2, ...
   *   └── dir3 - text1, text2, ...
   * Under the target path, there ought to be N subdirectories (dir1 to dirN). Each
   * subdirectory represents a category and contains all texts that belong to such
   * category. Each category will be a given a label according to its position in the
   * ascending order sorted among all subdirectories.
   * All texts will be given a label according to the subdirectory where it is located.
   * Labels start from 0.
   *
   * @param path The folder path to texts. Local or distributed file system (such as HDFS)
   *             are supported. If you want to read from a distributed file system, sc
   *             needs to be specified.
   * @param sc An instance of SparkContext.
   *           If specified, texts will be read as a DistributedTextSet.
   *           Default is null and in this case texts will be read as a LocalTextSet.
   * @param minPartitions Integer. A suggestion value of the minimal partition number for input
   *                      texts. Only need to specify this when sc is not null. Default is 1.
   * @return TextSet.
   */
  def read(path: String, sc: SparkContext = null, minPartitions: Int = 1): TextSet = {
    val textSet = if (sc != null) {
      // URI needs for the FileSystem to accept HDFS
      val uri = StringUtils.stringToURI(Array(path))(0)
      val fs = FileSystem.get(uri, new Configuration())
      val categories = fs.listStatus(new Path(path)).map(_.getPath.getName).sorted
      logger.info(s"Found ${categories.length} classes.")
      // Labels of categories start from 0.
      val indices = categories.indices
      val categoryToLabel = categories.zip(indices).toMap
      val textRDD = sc.wholeTextFiles(path + "/*", minPartitions).map{case (p, text) =>
        val parts = p.split("/")
        val category = parts(parts.length - 2)
        TextFeature(text, label = categoryToLabel(category), uri = p)
      }
      TextSet.rdd(textRDD)
    }
    else {
      val features = ArrayBuffer[TextFeature]()
      val categoryToLabel = new util.HashMap[String, Int]()
      val categoryPath = new File(path)
      require(categoryPath.exists(), s"$path doesn't exist. Please check your input path")
      val categoryPathList = categoryPath.listFiles().filter(_.isDirectory).toList.sorted
      categoryPathList.foreach { categoryPath =>
        val label = categoryToLabel.size()
        categoryToLabel.put(categoryPath.getName, label)
        val textFiles = categoryPath.listFiles()
          .filter(_.isFile).filter(_.getName.forall(Character.isDigit(_))).sorted
        textFiles.foreach { file =>
          val source = Source.fromFile(file, "ISO-8859-1")
          val text = try source.getLines().toList.mkString("\n") finally source.close()
          features.append(TextFeature(text, label, file.getAbsolutePath))
        }
      }
      logger.info(s"Found ${categoryToLabel.size()} classes")
      TextSet.array(features.toArray)
    }
    textSet
  }

  /**
   * Read texts with id from csv file.
   * Each record is supposed to contain the following two fields in order:
   * id(String) and text(String).
   *
   * @param path The path to the csv file. Local or distributed file system (such as HDFS)
   *             are supported. If you want to read from a distributed file system, sc
   *             needs to be specified.
   * @param sc An instance of SparkContext.
   *           If specified, texts will be read as a DistributedTextSet.
   *           Default is null and in this case texts will be read as a LocalTextSet.
   * @param minPartitions Integer. A suggestion value of the minimal partition number for input
   *                      texts. Only need to specify this when sc is not null. Default is 1.
   * @return TextSet.
   */
  def readCSV(path: String, sc: SparkContext = null, minPartitions: Int = 1): TextSet = {
    if (sc != null) {
      val textRDD = sc.textFile(path, minPartitions).map(line => {
        val subs = line.split(",", 2) // "," may exist in content.
        TextFeature(subs(1), uri = subs(0))
      })
      TextSet.rdd(textRDD)
    }
    else {
      val src = Source.fromFile(path)
      val textArray = src.getLines().toArray.map(line => {
        val subs = line.split(",", 2)
        TextFeature(subs(1), uri = subs(0))
      })
      TextSet.array(textArray)
    }
  }

  /**
   * Read texts with id from parquet file.
   * Schema should be the following:
   * "id"(String) and "text"(String).
   *
   * @param path The path to the parquet file.
   * @param sqlContext An instance of SQLContext.
   * @return DistributedTextSet.
   */
  def readParquet(path: String, sqlContext: SQLContext): DistributedTextSet = {
    val textRDD = sqlContext.read.parquet(path).rdd.map(row => {
      val uri = row.getAs[String]("id")
      val text = row.getAs[String](TextFeature.text)
      TextFeature(text, uri = uri)
    })
    TextSet.rdd(textRDD)
  }

  /**
   * Used to generate a TextSet for pairwise training.
   *
   * This method does the following:
   * 1. Generate all RelationPairs: (id1, id2Positive, id2Negative) from Relations.
   * 2. Join RelationPairs with corpus to transform id to indexedTokens.
   * Note: Make sure that the corpus has been transformed by [[SequenceShaper]] and [[WordIndexer]].
   * 3. For each pair, generate a TextFeature having Sample with:
   * - feature of shape (2, text1Length + text2Length).
   * - label of value [1 0] as the positive relation is placed before the negative one.
   *
   * @param relations RDD of [[Relation]].
   * @param corpus1 DistributedTextSet that contains all [[Relation.id1]]. For each TextFeature
   *                in corpus1, text must have been transformed to indexedTokens of the same length.
   * @param corpus2 DistributedTextSet that contains all [[Relation.id2]]. For each TextFeature
   *                in corpus2, text must have been transformed to indexedTokens of the same length.
   * @return DistributedTextSet.
   */
  def fromRelationPairs(
      relations: RDD[Relation],
      corpus1: TextSet,
      corpus2: TextSet,
      memoryType: MemoryType = DRAM): DistributedTextSet = {
    val pairsRDD = Relations.generateRelationPairs(relations)
    require(corpus1.isDistributed, "corpus1 must be a DistributedTextSet")
    require(corpus2.isDistributed, "corpus2 must be a DistributedTextSet")
    val joinedText1 = corpus1.toDistributed().rdd.keyBy(_.getURI)
      .join(pairsRDD.keyBy(_.id1)).map(_._2)
    val joinedText2Pos = corpus2.toDistributed().rdd.keyBy(_.getURI)
      .join(joinedText1.keyBy(_._2.id2Positive)).map(x => (x._2._2._1, x._2._1, x._2._2._2))
    val joinedText2Neg = corpus2.toDistributed().rdd.keyBy(_.getURI)
      .join(joinedText2Pos.keyBy(_._3.id2Negative))
      .map(x => (x._2._2._1, x._2._2._2, x._2._1))
    val res = joinedText2Neg.map(x => {
      val textFeature = TextFeature(null, x._1.getURI + x._2.getURI + x._3.getURI)
      val text1 = x._1.getIndices
      val text2Pos = x._2.getIndices
      val text2Neg = x._3.getIndices
      require(text1 != null,
        "corpus1 haven't been transformed from word to index yet, please word2idx first")
      require(text2Pos != null && text2Neg != null,
        "corpus2 haven't been transformed from word to index yet, please word2idx first")
      require(text2Pos.length == text2Neg.length,
        "corpus2 contains texts with different lengths, please shapeSequence first")
      val pairedIndices = text1 ++ text2Pos ++ text1 ++ text2Neg
      val feature = Tensor(pairedIndices, Array(2, text1.length + text2Pos.length))
      val label = Tensor(Array(1.0f, 0.0f), Array(2, 1))
      textFeature(TextFeature.sample) = Sample(feature, label)
      textFeature
    }).setName("Pairwise Training Set")
    TextSet.rdd(res, memoryType)
  }

  /**
   * Generate a TextSet for pairwise training using Relation array.
   *
   * @param relations Array of [[Relation]].
   * @param corpus1 LocalTextSet that contains all [[Relation.id1]]. For each TextFeature
   *                in corpus1, text must have been transformed to indexedTokens of the same length.
   * @param corpus2 LocalTextSet that contains all [[Relation.id2]]. For each TextFeature
   *                in corpus2, text must have been transformed to indexedTokens of the same length.
   * @return LocalTextSet.
   */
  def fromRelationPairs(
      relations: Array[Relation],
      corpus1: TextSet,
      corpus2: TextSet): LocalTextSet = {
    val pairsArray = Relations.generateRelationPairs(relations)
    require(corpus1.isLocal, "corpus1 must be a LocalTextSet")
    require(corpus2.isLocal, "corpus2 must be a LocalTextSet")
    val mapText1: MMap[String, Array[Float]] = MMap()
    val mapText2: MMap[String, Array[Float]] = MMap()
    val arrayText1 = corpus1.toLocal().array
    val arrayText2 = corpus2.toLocal().array
    for (text <- arrayText1) {
      val indices = text.getIndices
      require(indices != null,
        "corpus1 haven't been transformed from word to index yet, please word2idx first")
      mapText1(text.getURI) = indices
    }
    for (text <- arrayText2) {
      val indices = text.getIndices
      require(indices != null,
        "corpus2 haven't been transformed from word to index yet, please word2idx first")
      mapText2(text.getURI) = indices
    }
    val res = pairsArray.map(x => {
      val indices1 = mapText1(x.id1)
      val indices2Pos = mapText2(x.id2Positive)
      val indices2Neg = mapText2(x.id2Negative)
      require(indices2Neg.length == indices2Pos.length,
        "corpus2 contains texts with different lengths, please shapeSequence first")
      val textFeature = TextFeature(null, x.id1 + x.id2Positive + x.id2Negative)
      val pairedIndices = indices1 ++ indices2Pos ++ indices1 ++ indices2Neg
      val feature = Tensor(pairedIndices, Array(2, indices1.length + indices2Pos.length))
      val label = Tensor(Array(1.0f, 0.0f), Array(2, 1))
      textFeature(TextFeature.sample) = Sample(feature, label)
      textFeature
    })
    TextSet.array(res)
  }

  /**
   * Used to generate a TextSet for ranking.
   *
   * This method does the following:
   * 1. For each [[Relation.id1]], find the list of [[Relation.id2]] with corresponding
   * [[Relation.label]] that comes together with [[Relation.id1]].
   * In other words, group relations by [[Relation.id1]].
   * 2. Join with corpus to transform each id to indexedTokens.
   * Note: Make sure that the corpus has been transformed by [[SequenceShaper]] and [[WordIndexer]].
   * 3. For each list, generate a TextFeature having Sample with:
   * - feature of shape (listLength, text1Length + text2Length).
   * - label of shape (listLength, 1).
   *
   * @param relations RDD of [[Relation]].
   * @param corpus1 DistributedTextSet that contains all [[Relation.id1]]. For each TextFeature
   *                in corpus1, text must have been transformed to indexedTokens of the same length.
   * @param corpus2 DistributedTextSet that contains all [[Relation.id2]]. For each TextFeature
   *                in corpus2, text must have been transformed to indexedTokens of the same length.
   * @return DistributedTextSet.
   */
  def fromRelationLists(
      relations: RDD[Relation],
      corpus1: TextSet,
      corpus2: TextSet): DistributedTextSet = {
    require(corpus1.isDistributed, "corpus1 must be a DistributedTextSet")
    require(corpus2.isDistributed, "corpus2 must be a DistributedTextSet")
    val joinedText1 = corpus1.toDistributed().rdd.keyBy(_.getURI)
      .join(relations.keyBy(_.id1)).map(_._2)
    val joinedText2 = corpus2.toDistributed().rdd.keyBy(_.getURI).join(
      joinedText1.keyBy(_._2.id2))
      .map(x => (x._2._2._1, x._2._1, x._2._2._2.label))
    val joinedLists = joinedText2.groupBy(_._1.getURI).map(_._2.toArray)
    val res = joinedLists.map(x => {
      val text1 = x.head._1
      val text2Array = x.map(_._2)
      val textFeature = TextFeature(null,
        uri = text1.getURI ++ text2Array.map(_.getURI).mkString(""))
      val text1Indices = text1.getIndices
      require(text1Indices != null,
        "corpus1 haven't been transformed from word to index yet, please word2idx first")
      val text2IndicesArray = text2Array.map(_.getIndices)
      text2IndicesArray.foreach(x => require(x != null,
        "corpus2 haven't been transformed from word to index yet, please word2idx first"))
      val data = text2IndicesArray.flatMap(text1Indices ++ _)
      val feature = Tensor(data,
        Array(text2Array.length, text1Indices.length + text2IndicesArray.head.length))
      val label = Tensor(x.map(_._3.toFloat), Array(text2Array.length, 1))
      textFeature(TextFeature.sample) = Sample(feature, label)
      textFeature
    }).setName("Listwise Evaluation Set")
    TextSet.rdd(res)
  }

  /**
   * Generate a TextSet for ranking using Relation array.
   *
   * @param relations Array of [[Relation]].
   * @param corpus1 LocalTextSet that contains all [[Relation.id1]]. For each TextFeature
   *                in corpus1, text must have been transformed to indexedTokens of the same length.
   * @param corpus2 LocalTextSet that contains all [[Relation.id2]]. For each TextFeature
   *                in corpus2, text must have been transformed to indexedTokens of the same length.
   * @return LocalTextSet.
   */
  def fromRelationLists(
      relations: Array[Relation],
      corpus1: TextSet,
      corpus2: TextSet): LocalTextSet = {
    require(corpus1.isLocal, "corpus1 must be a LocalTextSet")
    require(corpus2.isLocal, "corpus2 must be a LocalTextSet")
    val mapText1: MMap[String, Array[Float]] = MMap()
    val mapText2: MMap[String, Array[Float]] = MMap()
    val arrayText1 = corpus1.toLocal().array
    val arrayText2 = corpus2.toLocal().array
    for (text <- arrayText1) {
      val indices = text.getIndices
      require(indices != null,
        "corpus1 haven't been transformed from word to index yet, please word2idx first")
      mapText1(text.getURI) = indices
    }
    for (text <- arrayText2) {
      val indices = text.getIndices
      require(indices != null,
        "corpus2 haven't been transformed from word to index yet, please word2idx first")
      mapText2(text.getURI) = indices
    }
    val text1Map: MMap[String, ArrayBuffer[(String, Int)]] = MMap()
    for (rel <- relations) {
      if (! text1Map.contains(rel.id1)) {
        val id2Array: ArrayBuffer[(String, Int)] = ArrayBuffer()
        id2Array.append((rel.id2, rel.label))
        text1Map(rel.id1) = id2Array
      }
      else {
        val id2Array = text1Map(rel.id1)
        id2Array.append((rel.id2, rel.label))
      }
    }
    val features: ArrayBuffer[TextFeature] = ArrayBuffer()
    for((id1, id2LabelArray) <- text1Map) {
      val id2ArrayLength = id2LabelArray.length
      val textFeature = TextFeature(null, uri = id1 ++ id2LabelArray.map(_._1).mkString(""))
      val indices2Array = id2LabelArray.map(x => {mapText2(x._1.toString)})
      val indices1 = mapText1(id1)
      val data = indices2Array.flatMap(indices1 ++ _).toArray
      val feature = Tensor(data,
        Array(id2ArrayLength, indices1.length + indices2Array.head.length))
      val label = Tensor(id2LabelArray.toArray.map(_._2.toFloat), Array(id2ArrayLength, 1))
      textFeature(TextFeature.sample) = Sample(feature, label)
      features.append(textFeature)
    }
    TextSet.array(features.toArray)
  }

  /**
   * Assign each word an index to form a map.
   *
   * @param words Array of words.
   * @param existingMap Existing map of word index if any. Default is null and in this case a new
   *                    map with index starting from 1 will be generated.
   *                    If not null, then the generated map will preserve the word index in
   *                    existingMap and assign subsequent indices to new words.
   * @return wordIndex map.
   */
  def wordsToMap(words: Array[String], existingMap: Map[String, Int] = null): Map[String, Int] = {
    if (existingMap == null) {
      val indexes = Range(1, words.length + 1)
      words.zip(indexes).map{item =>
        (item._1, item._2)}.toMap
    }
    else {
      val resMap = collection.mutable.Map(existingMap.toSeq: _*)
      var i = existingMap.values.max + 1
      for (word <- words) {
        if (!existingMap.contains(word)) {
          resMap(word) = i
          i += 1
        }
      }
      resMap.toMap
    }
  }
}


/**
 * LocalTextSet is comprised of array of TextFeature.
 */
class LocalTextSet(var array: Array[TextFeature]) extends TextSet {

  override def transform(transformer: Preprocessing[TextFeature, TextFeature]): TextSet = {
    array = transformer.apply(array.toIterator).toArray
    this
  }

  override def isLocal: Boolean = true

  override def isDistributed: Boolean = false

  override def toLocal(): LocalTextSet = {
    this
  }

  override def toDistributed(sc: SparkContext, partitionNum: Int = 4): DistributedTextSet = {
    new DistributedTextSet(sc.parallelize(array, partitionNum))
  }

  override def toDataSet: DataSet[Sample[Float]] = {
    DataSet.array(array.map(_[Sample[Float]](TextFeature.sample)))
  }

  override def randomSplit(weights: Array[Double]): Array[TextSet] = {
    throw new UnsupportedOperationException("LocalTextSet doesn't support randomSplit for now")
  }

  override def generateWordIndexMap(
    removeTopN: Int = 0,
    maxWordsNum: Int = -1,
    minFreq: Int = 1,
    existingMap: Map[String, Int] = null): Map[String, Int] = {
    require(removeTopN >= 0, "removeTopN should be a non-negative integer")
    require(maxWordsNum == -1 || maxWordsNum > 0,
      "maxWordsNum should be either -1 or a positive integer")
    require(minFreq >= 1, "minFreq should be a positive integer")
    val words = if (removeTopN == 0 && maxWordsNum == -1 && minFreq == 1) {
      array.flatMap(_.getTokens).distinct
    }
    else {
      var frequencies = array.flatMap(_.getTokens)
        .groupBy(identity).mapValues(_.length).toArray
      if (minFreq > 1) frequencies = frequencies.filter(_._2 >= minFreq)
      if (removeTopN > 0 || maxWordsNum > 0) {
        // Need to sort by frequency in this case.
        var res = frequencies.sortBy(- _._2).map(_._1)
        if (removeTopN > 0) res = res.drop(removeTopN)
        if (maxWordsNum > 0) res = res.take(maxWordsNum)
        res
      }
      else frequencies.map(_._1)
    }
    val wordIndex = TextSet.wordsToMap(words, existingMap)
    setWordIndex(wordIndex)
    wordIndex
  }

  override def saveWordIndex(path: String): Unit = {
    super.saveWordIndex(path)
    val pw = new PrintWriter(new File(path))
    for (item <- getWordIndex) {
      pw.print(item._1)
      pw.print(" ")
      pw.println(item._2)
    }
    pw.close()
  }

  override def loadWordIndex(path: String): TextSet = {
    val wordIndex = MMap[String, Int]()
    for (line <- Source.fromFile(path).getLines) {
      val values = line.split(" ")
      wordIndex.put(values(0), values(1).toInt)
    }
    setWordIndex(wordIndex.toMap)
  }
}


/**
 * DistributedTextSet is comprised of RDD of TextFeature.
 */
class DistributedTextSet(var rdd: RDD[TextFeature],
                         memoryType: MemoryType = DRAM) extends TextSet {

  override def transform(transformer: Preprocessing[TextFeature, TextFeature]): TextSet = {
    rdd = transformer(rdd)
    this
  }

  override def isLocal: Boolean = false

  override def isDistributed: Boolean = true

  override def toLocal(): LocalTextSet = {
    new LocalTextSet(rdd.collect())
  }

  override def toDistributed(sc: SparkContext = null, partitionNum: Int = 4): DistributedTextSet = {
    this
  }

  override def toDataSet: DataSet[Sample[Float]] = {
    FeatureSet.rdd(rdd.map(_[Sample[Float]](TextFeature.sample))
      .setName(s"Samples in ${rdd.name}"),
      memoryType)
  }

  override def randomSplit(weights: Array[Double]): Array[TextSet] = {
    rdd.randomSplit(weights).map(v => TextSet.rdd(v, this.memoryType))
  }

  override def generateWordIndexMap(
    removeTopN: Int = 0,
    maxWordsNum: Int = -1,
    minFreq: Int = 1,
    existingMap: Map[String, Int] = null): Map[String, Int] = {
    require(removeTopN >= 0, "removeTopN should be a non-negative integer")
    require(maxWordsNum == -1 || maxWordsNum > 0,
      "maxWordsNum should be either -1 or a positive integer")
    require(minFreq >= 1, "minFreq should be a positive integer")
    val words = if (removeTopN == 0 && maxWordsNum == -1 && minFreq == 1) {
      rdd.flatMap(_.getTokens).distinct().collect()
    }
    else {
      var frequencies = rdd.flatMap(_.getTokens)
        .map(word => (word, 1)).reduceByKey(_ + _)
      if (minFreq > 1) frequencies = frequencies.filter(_._2 >= minFreq)
      if (removeTopN > 0 || maxWordsNum > 0) {
        // Need to sort by frequency in this case
        var res = frequencies.sortBy(- _._2).map(_._1).collect()
        if (removeTopN > 0) res = res.drop(removeTopN)
        if (maxWordsNum > 0) res = res.take(maxWordsNum)
        res
      }
      else frequencies.map(_._1).collect()
    }
    val wordIndex = TextSet.wordsToMap(words, existingMap)
    setWordIndex(wordIndex)
    wordIndex
  }

  override def saveWordIndex(path: String): Unit = {
    super.saveWordIndex(path)
    val fs = FileSystem.get(rdd.sparkContext.hadoopConfiguration)
    val os = new BufferedOutputStream(fs.create(new Path(path)))
    for (item <- getWordIndex) {
      os.write((item._1 + " " + item._2 + "\n").getBytes("UTF-8"))
    }
    os.close()
  }

  override def loadWordIndex(path: String): TextSet = {
    val fs = FileSystem.get(rdd.sparkContext.hadoopConfiguration)
    val br = new BufferedReader(new InputStreamReader(fs.open(new Path(path))))
    val wordIndex = MMap[String, Int]()
    var line = br.readLine()
    while (line != null) {
      val values = line.split(" ")
      wordIndex.put(values(0), values(1).toInt)
      line = br.readLine()
    }
    setWordIndex(wordIndex.toMap)
  }
}
