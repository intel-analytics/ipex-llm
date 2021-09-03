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

package com.intel.analytics.zoo.pipeline.api.keras.layers

import java.io._
import java.util.concurrent.atomic.AtomicInteger

import com.intel.analytics.bigdl.nn.{Identity => BIdentity}
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.serialization.Bigdl._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.utils.serializer.{DeserializeContext, SerializeContext}
import com.intel.analytics.bigdl.utils.serializer.converters.{DataConverter, TensorConverter}
import com.intel.analytics.zoo.common.CheckedObjectInputStream
import com.intel.analytics.zoo.pipeline.api.keras.layers.WordEmbedding.EmbeddingMatrixHolder
import com.intel.analytics.zoo.pipeline.api.net.{NetUtils, RegistryMap, SerializationHolder}
import org.slf4j.LoggerFactory

import scala.collection.mutable.{Map => MMap}
import scala.io.Source
import scala.reflect.ClassTag
import scala.reflect.runtime.universe

/**
 * Embedding layer that directly loads pre-trained word vectors as weights.
 * Turn non-negative integers (indices) into dense vectors of fixed size.
 * Currently only GloVe embedding is supported.
 * The input of this layer should be 2D.
 *
 * This layer can only be used as the first layer in a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 */
class WordEmbedding[T: ClassTag] private(
    override val inputDim: Int,
    override val outputDim: Int,
    embeddingMatrix: EmbeddingMatrixHolder[T],
    override val trainable: Boolean = false,
    override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends Embedding[T](inputDim, outputDim, inputShape = inputShape) {

  require(!trainable, "WordEmbedding is not trainable for now.")

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    BIdentity().asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }

  private def weight: Tensor[T] = embeddingMatrix.weight

  private val inputBuffer = Tensor[T]()

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    inputBuffer.resizeAs(input.contiguous())
    inputBuffer.fill(ev.one).add(input)
    try {
      output.index(1, inputBuffer.view(inputBuffer.nElement()), weight)
      output = output.view(inputBuffer.size(1), inputBuffer.size(2), weight.size(2))
    } catch {
      case e: IllegalArgumentException =>
        throw new IllegalArgumentException(
          s"EmbeddingGloVe updateOutput get exception: ${e.getMessage}\n" +
            s"please ensure all elements of your input smaller than $inputDim.", e)
      case e: Exception =>
        throw e
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (!gradInput.isSameSizeAs(input)) {
      gradInput.resizeAs(input).zero()
    }
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(), Array())
  }

  override def clearState(): this.type = {
    super.clearState()
    inputBuffer.set()
    this
  }
}

object WordEmbedding {

  DataConverter.registerConverter(
    "com.intel.analytics.zoo.pipeline.api.keras.layers.WordEmbedding.EmbeddingMatrixHolder[T]",
    EmbeddingMatrixHolderConverter)

  val id = new AtomicInteger(0) // id in the registry map should be unique

  /**
   * Embedding layer that directly loads pre-trained word vectors as weights.
   * Please use this layer as the first layer in a model.
   *
   * @param embeddingFile The path to the embedding file.
   *                      Currently the following GloVe files are supported:
   *                      "glove.6B.50d.txt", "glove.6B.100d.txt", "glove.6B.200d.txt"
   *                      "glove.6B.300d.txt", "glove.42B.300d.txt", "glove.840B.300d.txt".
   *                      You can download them from: https://nlp.stanford.edu/projects/glove/.
   * @param wordIndex Map of word (String) and its corresponding index (integer).
   *                  The index is supposed to start from 1 with 0 reserved for unknown words.
   *                  During the prediction, if you have words that are not in the wordIndex
   *                  for the training, you can map them to index 0.
   *                  Default is null. In this case, all the words in the embeddingFile will
   *                  be taken into account and you can call
   *                  WordEmbedding.getWordIndex(embeddingFile) to retrieve the map.
   * @param trainable To configure whether the weights of this layer will be updated or not.
   *                  Only false is supported for now.
   * @param inputLength Positive integer. The sequence length of each input.
   * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
   */
  def apply[@specialized(Float, Double) T: ClassTag](
      embeddingFile: String,
      wordIndex: Map[String, Int] = null,
      trainable: Boolean = false,
      inputLength: Int = -1)(implicit ev: TensorNumeric[T]): WordEmbedding[T] = {
    val shape = if (inputLength > 0) Shape(inputLength) else null
    val (inputDim, outputDim, embeddingMatrix) = prepareEmbedding[T](embeddingFile, wordIndex)
    id.getAndIncrement()
    new WordEmbedding[T](inputDim, outputDim,
      new EmbeddingMatrixHolder[T](embeddingMatrix, "WordEmbedding" + id.toString),
      trainable, shape)
  }

  /**
   * Prepare embedding weights from embeddingFile given wordIndex.
   *
   * @param randomizeUnknown Boolean. Whether to randomly initialize words that don't exist in
   *                         embedding_file. Default is false and in this case corresponding entries
   *                         to unknown words will be zero vectors.
   * @param normalize Boolean. Whether to normalize word vectors. Default is false.
   * @return Embedding input dim, output dim and pretrained weights.
   */
  def prepareEmbedding[@specialized(Float, Double) T: ClassTag](
      embeddingFile: String,
      wordIndex: Map[String, Int],
      randomizeUnknown: Boolean = false,
      normalize: Boolean = false)
    (implicit ev: TensorNumeric[T]): (Int, Int, Tensor[T]) = {
    require(new File(embeddingFile).exists(),
      s"embeddingFile $embeddingFile doesn't exist. Please check your file path.")

    if (wordIndex != null) {
      require(wordIndex.values.forall(_ > 0),
        "In wordIndex, indices should be positive and start from 1 " +
          "with 0 reserved for unknown words.")
    }

    val indexVec = buildIndexVec[T](wordIndex, embeddingFile)
    val inputDim = if (wordIndex == null) calcInputDimFromIndexVec[T](indexVec)
    else calcInputDimFromWordIndex(wordIndex)
    val outputDim = getOutputDimFromEmbeddingFile(embeddingFile)
    val embeddingMatrix = buildEmbeddingMatrix[T](indexVec, inputDim,
      outputDim, randomizeUnknown, normalize)
    (inputDim, outputDim, embeddingMatrix)
  }

  def calcInputDimFromWordIndex(wordIndex: Map[String, Int]): Int = {
    // Use max instead of length here in case the indices are not continuous.
    // +1 for unknown index 0.
    wordIndex.values.max + 1
  }

  def calcInputDimFromIndexVec[@specialized(Float, Double) T: ClassTag](
      indexVec: Map[Int, Array[T]])(implicit ev: TensorNumeric[T]): Int = {
    // +1 for unknown index 0.
    indexVec.keys.max + 1
  }

  def getOutputDimFromEmbeddingFile(embeddingFile: String): Int = {
    embeddingFile.split("/").last match {
      case "glove.6B.50d.txt" => 50
      case "glove.6B.100d.txt" => 100
      case "glove.6B.200d.txt" => 200
      case "glove.6B.300d.txt" => 300
      case "glove.42B.300d.txt" => 300
      case "glove.840B.300d.txt" => 300
      case _ => throw new IllegalArgumentException(s"Unsupported embeddingFile: " +
        s"$embeddingFile")
    }
  }

  def buildIndexVec[@specialized(Float, Double) T: ClassTag](
      wordIndex: Map[String, Int],
      embeddingFile: String)(implicit ev: TensorNumeric[T]): Map[Int, Array[T]] = {
    if (wordIndex == null) {
      buildFullEmbedding[T](embeddingFile)._2
    }
    else {
      indexVecFromWordIndex[T](wordIndex, embeddingFile)
    }
  }

  // return wordIndex map and indexVec map for the whole embedding file
  def buildFullEmbedding[@specialized(Float, Double) T: ClassTag](
      embeddingFile: String)(implicit ev: TensorNumeric[T]):
      (Map[String, Int], Map[Int, Array[T]]) = {
    logger.debug(s"Indexing all word vectors in $embeddingFile.")
    val wordIndex = MMap[String, Int]()
    val indexVec = MMap[Int, Array[T]]()
    var i = 0
    for (line <- Source.fromFile(embeddingFile, "ISO-8859-1").getLines) {
      i += 1
      val values = line.split(" ")
      val word = values(0)
      val vector = values.slice(1, values.length).map(v => ev.fromType(v.toFloat))
      wordIndex.put(word, i)
      indexVec.put(i, vector)
    }
    logger.debug(s"There are totally ${indexVec.size} word vectors in $embeddingFile.")
    (wordIndex.toMap, indexVec.toMap)
  }

  def indexVecFromWordIndex[@specialized(Float, Double) T: ClassTag](
      wordIndex: Map[String, Int],
      embeddingFile: String)(implicit ev: TensorNumeric[T]): Map[Int, Array[T]] = {
    logger.debug(s"Indexing word vectors in $embeddingFile.")
    val indexVec = MMap[Int, Array[T]]()
    for (line <- Source.fromFile(embeddingFile, "ISO-8859-1").getLines) {
      val values = line.split(" ")
      val word = values(0)
      if (wordIndex.keySet.contains(word)) {
        val vector = values.slice(1, values.length).map(v => ev.fromType(v.toFloat))
        indexVec.put(wordIndex(word), vector)
      }
    }
    logger.debug(s"Found ${indexVec.size} word vectors existing in $embeddingFile.")
    indexVec.toMap
  }

  def buildEmbeddingMatrix[@specialized(Float, Double) T: ClassTag](
      indexVec: Map[Int, Array[T]],
      inputDim: Int,
      outputDim: Int,
      randomizeUnknown: Boolean = false,
      normalize: Boolean = false)(implicit ev: TensorNumeric[T]): Tensor[T] = {
    val weights = Tensor[T](inputDim, outputDim).zero()
    for (i <- 1 until inputDim) {
      if (indexVec.get(i).isDefined) {
        val vec = if (normalize) normalizeVector(indexVec(i)) else indexVec(i)
        weights.narrow(1, i + 1, 1).copy(Tensor[T](vec, Array(outputDim)))
      }
      else if (randomizeUnknown) {
        val vec = if (normalize) normalizeVector(randomVector(outputDim))
        else randomVector(outputDim)
        weights.narrow(1, i + 1, 1).copy(Tensor[T](vec, Array(outputDim)))
      }
    }
    weights
  }

  def normalizeVector[@specialized(Float, Double) T: ClassTag](
      vec: Array[T])(implicit ev: TensorNumeric[T]): Array[T] = {
    val sum = ev.sqrt(ev.sum(vec.length, vec.map(x => ev.times(x, x)), 0, 1))
    vec.map(ev.divide(_, sum))
  }

  def randomVector[@specialized(Float, Double) T: ClassTag](
      dim: Int)(implicit ev: TensorNumeric[T]): Array[T] = {
    val alpha = ev.rand()
    val vector = Array.fill(dim) {1}.map(x => {
      ev.times(ev.minus(ev.times(ev.fromType(2.0), ev.rand()), ev.fromType(1.0)), alpha)
    })
    vector
  }

  /**
   * Get the full wordIndex map from the given embeddingFile.
   *
   * @param embeddingFile The path to the embedding file.
   *                      Currently only the following GloVe files are supported:
   *                      "glove.6B.50d.txt", "glove.6B.100d.txt", "glove.6B.200d.txt"
   *                      "glove.6B.300d.txt", "glove.42B.300d.txt", "glove.840B.300d.txt".
   *                      You can download them from: https://nlp.stanford.edu/projects/glove/.
   * @return Map of word (String) and its corresponding index (integer) obtained from
   *         the given embedding file.
   */
  def getWordIndex(embeddingFile: String): Map[String, Int] = {
    buildFullEmbedding[Float](embeddingFile)._1
  }

  @transient
  private lazy val inDriver = NetUtils.isDriver

  private val logger = LoggerFactory.getLogger(getClass)

  private val weightRegistry = new RegistryMap[Tensor[_]]()

  class EmbeddingMatrixHolder[T: ClassTag](
      @transient var weight: Tensor[T],
      private var id: String)(implicit ev: TensorNumeric[T])
    extends SerializationHolder {

    def getId: String = id

    override def writeInternal(out: CommonOutputStream): Unit = {
      val (cachedWeight, _) = weightRegistry.getOrCreate(id) {
        timing("exporting as weight tensor") {
          weight
        }
      }
      out.writeString(id)
      if (inDriver) {
        val stream = new ByteArrayOutputStream()
        val oos = new ObjectOutputStream(stream)
        oos.writeObject(cachedWeight)
        oos.close()
        val w = stream.toByteArray
        val len = w.length
        out.writeInt(len)
        timing(s"writing ${len / 1024 / 1024}Mb weight to stream") {
          out.write(w)
        }
      }
      else {
        out.writeInt(0)
      }
    }

    override def readInternal(in: CommonInputStream): Unit = {
      id = in.readString()
      val (cachedWeight, isCreated) = weightRegistry.getOrCreate(id) {
        val len = in.readInt()
        require(len != 0, "weight length should not be zero," +
          "please set logging level to debug for more information")
        assert(len >= 0, "weight length should be an non-negative integer")
        val w = new Array[Byte](len)
        timing("reading weight from stream") {
          var numOfBytes = 0
          while (numOfBytes < len) {
            val read = in.read(w, numOfBytes, len - numOfBytes)
            numOfBytes += read
          }
        }
        val ois = new CheckedObjectInputStream(classOf[Tensor[T]], new ByteArrayInputStream(w))
        try {
          ois.readObject().asInstanceOf[Tensor[T]]
        } finally {
          ois.close()
        }
      }
      if (!isCreated) {
        val len = in.readInt()
        assert(len >= 0, "weight length should be an non-negative integer")
        in.skip(len)
      }
      weight = cachedWeight.asInstanceOf[Tensor[T]]
    }
  }
}


object EmbeddingMatrixHolderConverter extends DataConverter {
  override def getAttributeValue[T: ClassTag](
      context: DeserializeContext,
      attribute: AttrValue)(implicit ev: TensorNumeric[T]): AnyRef = {
    val map = attribute.getNameAttrListValue.getAttrMap
    val id = map.get("id").getStringValue
    val weight = TensorConverter.getAttributeValue[T](context, map.get("weight"))
    new EmbeddingMatrixHolder[T](weight.asInstanceOf[Tensor[T]], id)
  }

  override def setAttributeValue[T: ClassTag](
      context: SerializeContext[T],
      attributeBuilder: AttrValue.Builder,
      value: Any,
      valueType: universe.Type = null)(implicit ev: TensorNumeric[T]): Unit = {
    val matrixHolder = value.asInstanceOf[WordEmbedding.EmbeddingMatrixHolder[T]]
    val idAttr = AttrValue.newBuilder()
      .setDataType(DataType.STRING)
      .setStringValue(matrixHolder.getId).build()
    val weightAttr = AttrValue.newBuilder
    TensorConverter.setAttributeValue(context, weightAttr, matrixHolder.weight)
    val attrList = NameAttrList.newBuilder()
      .putAttr("id", idAttr).putAttr("weight", weightAttr.build())
    attributeBuilder.setNameAttrListValue(attrList)
  }
}
