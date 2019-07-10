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

import com.intel.analytics.bigdl.nn.{RandomNormal, StaticGraph}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.keras.{KerasLayer, KerasLayerSerializable}
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer._
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import com.intel.analytics.bigdl.utils.{MultiShape, Shape}
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.autograd.{AutoGrad, Variable}
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.{GraphRef, KerasUtils}
import com.intel.analytics.zoo.pipeline.api.keras.models.Model
import com.intel.analytics.zoo.pipeline.api.keras.models.Model.{apply => _, _}
import org.apache.log4j.Logger

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.reflect.runtime._

/**
 * [[BERT]] A self attention keras like layer.
 * Input is a Table which consists of 4 tensors.
 * 1. Token id tensor: shape [batch, seqLen] with the word token indices in the vocabulary
 * 2. Token type id tensor: shape [batch, seqLen] with the token types in [0, 1].
 *    0 means `sentence A` and 1 means a `sentence B` (see BERT paper for more details).
 * 3. Position id tensor: shape [batch, seqLen] with positions in the sentence.
 * 4. Attention_mask tensor: shape [batch, seqLen] with indices in [0, 1].
 *   It's a mask to be used if the input sequence length is smaller than seqLen in
 *   the current batch.
 * Output is an Activity which output the states of BERT layer
 * @param nBlock block number
 * @param nHead head number
 * @param intermediateSize The size of the "intermediate" (i.e., feed-forward)
 * @param hiddenPDrop The dropout probability for all fully connected layers
 * @param attnPDrop drop probability of attention
 * @param initializerRange weight initialization range
 * @param outputAllBlock whether output all blocks' output
 * @param embeddingLayer embedding layer
 * @param inputShape input shape, default is null
 */
class BERT[T: ClassTag] private (
  nBlock: Int = 12,
  nHead: Int = 12,
  intermediateSize: Int = 3072,
  hiddenPDrop: Double = 0.1,
  attnPDrop: Double = 0.1,
  initializerRange: Double = 0.02,
  outputAllBlock: Boolean = true,
  embeddingLayer: KerasLayer[Activity, Tensor[T], T],
  inputShape: Shape)(implicit ev: TensorNumeric[T])
  extends TransformerLayer[T](nBlock, hiddenPDrop, attnPDrop, nHead,
    initializerRange, true, outputAllBlock, embeddingLayer, intermediateSize, inputShape)
  with Net {

  private var vocab: Int = 0
  private var hiddenSize: Int = 0
  private var maxPositionLen: Int = 0

  override def projectionLayer(outputSize: Int): Net = {
    new Dense(outputSize, init = RandomNormal(0.0, initializerRange))
  }

  override def gelu(x: Variable[T]): Variable[T] = {
    val y = x / math.sqrt(2.0)
    val e = AutoGrad.erf(y)
    x * 0.5 * (e + 1.0)
  }

  override def buildInput(inputShape: Shape):
  (Variable[T], List[Variable[T]], List[Variable[T]]) = {
    require(inputShape.isInstanceOf[MultiShape] &&
      inputShape.asInstanceOf[MultiShape].value.size == 4, "BERT input must be" +
      " a list of 4 tensors (consisting of input sequence, sequence positions," +
      "segment id, attention mask)")
    val _inputShape = KerasUtils.removeBatch(inputShape).toMulti()
    seqLen = _inputShape.head.toSingle().head

    val inputs = _inputShape.map(Variable(_))
    return ((- inputs.last + 1.0) * -10000.0, inputs.dropRight(1), inputs)
  }

  override def allowRebuilt: Boolean = true
}

object BERT extends KerasLayerSerializable {
  Model
  ModuleSerializer.registerModule(
    "com.intel.analytics.zoo.pipeline.api.keras.layers.BERT",
    BERT)

  private val logger = Logger.getLogger(getClass)

  /**
   * [[BERT]] A self attention keras like layer
   * @param vocab vocabulary size of training data, default is 40990
   * @param hiddenSize size of the encoder layers, default is 768
   * @param nBlock block number, default is 12
   * @param nHead head number, default is 12
   * @param maxPositionLen sequence length, default is 512
   * @param intermediateSize The size of the "intermediate" (i.e., feed-forward), default is 3072
   * @param hiddenPDrop The dropout probability for all fully connected layers, default is 0.1
   * @param attnPDrop drop probability of attention, default is 0.1
   * @param initializerRange weight initialization range, default is 0.02
   * @param outputAllBlock whether output all blocks' output, default is true
   * @param inputSeqLen sequence length of input, default is -1
   *                    which means the same with maxPositionLen
   */
  def apply[@specialized(Float, Double) T: ClassTag](
    vocab: Int = 40990,
    hiddenSize: Int = 768,
    nBlock: Int = 12,
    nHead: Int = 12,
    maxPositionLen: Int = 512,
    intermediateSize: Int = 3072,
    hiddenPDrop: Double = 0.1,
    attnPDrop: Double = 0.1,
    initializerRange: Double = 0.02,
    outputAllBlock: Boolean = true,
    inputSeqLen: Int = -1
    )(implicit ev: TensorNumeric[T]): BERT[T] = {
    require(hiddenSize > 0, "hiddenSize must be great" +
      "than 0 with default embedding layer")
    val len = if (inputSeqLen > 0) inputSeqLen else maxPositionLen
    val wordInput = Variable(Shape(len))
    val tokenTypeInput = Variable(Shape(len))
    val positionInput = Variable(Shape(len))
    val initWordEmbeddingW = Tensor[T](vocab, hiddenSize).randn(0.0, initializerRange)
    val initPositionEmbeddingW = Tensor[T](maxPositionLen, hiddenSize).randn(0.0, initializerRange)
    val initTokenEmbeddingW = Tensor[T](2, hiddenSize).randn(0.0, initializerRange)
    val wordEmbeddings = new Embedding(vocab, hiddenSize,
      initWeights = initWordEmbeddingW).from(wordInput)
    val positionEmbeddings = new Embedding(maxPositionLen, hiddenSize,
      initWeights = initPositionEmbeddingW).from(positionInput)
    val tokenTypeEmbeddings = new Embedding(2, hiddenSize,
      initWeights = initTokenEmbeddingW).from(tokenTypeInput)

    val embeddings = wordEmbeddings + positionEmbeddings + tokenTypeEmbeddings
    val afterNorm = LayerNorm[T](nOutput = hiddenSize, eps = 1e-12).from(embeddings)
    val h = Dropout(hiddenPDrop).from(afterNorm)

    val embeddingLayer = Model(Array(wordInput, tokenTypeInput, positionInput), h)
    val bert = new BERT[T](nBlock, nHead, intermediateSize, hiddenPDrop, attnPDrop,
      initializerRange, outputAllBlock,
      embeddingLayer.asInstanceOf[KerasLayer[Activity, Tensor[T], T]], null)
    bert.vocab = vocab
    bert.hiddenSize = hiddenSize
    bert.maxPositionLen = maxPositionLen
    bert
  }

  /**
   * [[BERT]] A self attention keras like layer
   * @param nBlock block number
   * @param nHead head number
   * @param intermediateSize The size of the "intermediate" (i.e., feed-forward)
   * @param hiddenPDrop The dropout probability for all fully connected layers
   * @param attnPDrop drop probability of attention
   * @param initializerRange weight initialization range
   * @param outputAllBlock whether output all blocks' output
   * @param embeddingLayer embedding layer
   */
  def apply[@specialized(Float, Double) T: ClassTag](
    nBlock: Int,
    nHead: Int,
    intermediateSize: Int,
    hiddenPDrop: Double,
    attnPDrop: Double,
    initializerRange: Double,
    outputAllBlock: Boolean,
    embeddingLayer: KerasLayer[Activity, Tensor[T], T])
    (implicit ev: TensorNumeric[T]): BERT[T] = {
    new BERT[T](nBlock, nHead, intermediateSize, hiddenPDrop, attnPDrop, initializerRange,
      outputAllBlock, embeddingLayer, null)
  }

  /**
   * create BERT from an existing model (with weights).
   *
   * @param path The path for the pre-defined model.
   *             Local file system, HDFS and Amazon S3 are supported.
   *             HDFS path should be like "hdfs://[host]:[port]/xxx".
   *             Amazon S3 path should be like "s3a://bucket/xxx".
   * @param weightPath The path for pre-trained weights if any.
   * @param inputSeqLen sequence length of input, will be ignored if existing model is built with
   *                   customized embedding
   * @param hiddenPDrop The dropout probability for all fully connected layers, will be
   *                   ignored if existing model is built with customized embedding
   * @param attnPDrop drop probability of attention, will be ignored if existing model is
   *                  built with customized embedding
   */
  def apply[T: ClassTag](path: String, weightPath: String, inputSeqLen: Int,
    hiddenPDrop: Double, attnPDrop: Double, outputAllBlock: Boolean)
    (implicit ev: TensorNumeric[T]): BERT[T] = {
    val loadedModel = ModuleLoader.loadFromFile(path, weightPath).asInstanceOf[BERT[T]]
    if (loadedModel.vocab == 0) {
      logger.warn("Configuration of inputSeqLen/hiddenPDrop/attnPDrop/outputAllBlock will be" +
        "ignored since pretrained model is built with customized embedding layer.")
      return loadedModel
    }

    val seqLength = if (inputSeqLen < 0) loadedModel.seqLen else inputSeqLen
    val hDrop = if (hiddenPDrop < 0) loadedModel.hiddenPDrop else hiddenPDrop
    val aDrop = if (attnPDrop < 0) loadedModel.attnPDrop else attnPDrop
    val newModel = BERT(loadedModel.vocab, loadedModel.hiddenSize, loadedModel.nBlock,
      loadedModel.nHead, loadedModel.maxPositionLen, loadedModel.intermediateSize, hDrop,
      aDrop, loadedModel.initializerRange, outputAllBlock, seqLength)
    val shape = Shape(List(Shape(seqLength), Shape(seqLength), Shape(seqLength),
      Shape(1, 1, seqLength)))
    newModel.build(KerasUtils.addBatch(shape))
    val parameter = loadedModel.parameters()._1
    val newParameter = newModel.parameters()._1
    var i = 0
    while(i < parameter.length) {
      newParameter(i).set(parameter(i))
      i += 1
    }
    newModel
  }

  override def doLoadModule[T: ClassTag](context : DeserializeContext)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {
    val attrMap = context.bigdlModule.getAttrMap

    val vocabAttr = attrMap.get("vocab")
    val vocab =
      DataConverter.getAttributeValue(context, vocabAttr)
        .asInstanceOf[Int]

    val hiddenSizeAttr = attrMap.get("hiddenSize")
    val hiddenSize =
      DataConverter.getAttributeValue(context, hiddenSizeAttr)
        .asInstanceOf[Int]

    val nBlockAttr = attrMap.get("nBlock")
    val nBlock =
      DataConverter.getAttributeValue(context, nBlockAttr)
        .asInstanceOf[Int]

    val nHeadAttr = attrMap.get("nHead")
    val nHead =
      DataConverter.getAttributeValue(context, nHeadAttr)
        .asInstanceOf[Int]

    val intermediateSizeAttr = attrMap.get("intermediateSize")
    val intermediateSize =
      DataConverter.getAttributeValue(context, intermediateSizeAttr)
        .asInstanceOf[Int]

    val hiddenPDropAttr = attrMap.get("hiddenPDrop")
    val hiddenPDrop =
      DataConverter.getAttributeValue(context, hiddenPDropAttr)
        .asInstanceOf[Double]

    val attnPDropAttr = attrMap.get("attnPDrop")
    val attnPDrop =
      DataConverter.getAttributeValue(context, attnPDropAttr)
        .asInstanceOf[Double]

    val initializerRangeAttr = attrMap.get("initializerRange")
    val initializerRange =
      DataConverter.getAttributeValue(context, initializerRangeAttr)
        .asInstanceOf[Double]

    val outputAllBlockAttr = attrMap.get("outputAllBlock")
    val outputAllBlock =
      DataConverter.getAttributeValue(context, outputAllBlockAttr)
        .asInstanceOf[Boolean]

    import scala.collection.JavaConverters._
    val subProtoModules = context.bigdlModule.getSubModulesList.asScala
    val subModules = subProtoModules.map(module => {
      val subModuleData = ModuleSerializer.load(DeserializeContext(module,
        context.storages, context.storageType, _copyWeightAndBias))
      subModuleData.module
    })
    val tGraph = subModules(0).asInstanceOf[StaticGraph[T]]
    val embeddingLayer = Model(tGraph.inputs.toArray, new GraphRef(tGraph).getOutputs().toArray)

    val seqLenAttr = attrMap.get("seqLen")
    val seqLen = DataConverter.getAttributeValue(context, seqLenAttr).asInstanceOf[Int]

    val maxPositionLenAttr = attrMap.get("maxPositionLen")
    val maxPositionLen =
      DataConverter.getAttributeValue(context, maxPositionLenAttr).asInstanceOf[Int]

    val bert = new BERT[T](nBlock, nHead, intermediateSize, hiddenPDrop, attnPDrop,
      initializerRange, outputAllBlock,
      embeddingLayer.asInstanceOf[KerasLayer[Activity, Tensor[T], T]], null)
    val tGraph2 = subModules(1).asInstanceOf[StaticGraph[T]]
    val labor = Model(tGraph2.inputs.toArray, new GraphRef(tGraph2).getOutputs().toArray)
    bert.labor = labor
    bert.seqLen = seqLen
    bert.vocab = vocab
    bert.hiddenSize = hiddenSize
    bert.maxPositionLen = maxPositionLen

    bert.asInstanceOf[AbstractModule[Activity, Activity, T]]
  }

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
    bertBuilder : BigDLModule.Builder)
    (implicit ev: TensorNumeric[T]) : Unit = {

    val bert = context.moduleData.module.asInstanceOf[BERT[T]]

    val vocabBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, vocabBuilder,
      bert.vocab, universe.typeOf[Int])
    bertBuilder.putAttr("vocab", vocabBuilder.build)

    val hiddenSizeBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, hiddenSizeBuilder,
      bert.hiddenSize, universe.typeOf[Int])
    bertBuilder.putAttr("hiddenSize", hiddenSizeBuilder.build)

    val nBlockBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, nBlockBuilder,
      bert.nBlock, universe.typeOf[Int])
    bertBuilder.putAttr("nBlock", nBlockBuilder.build)

    val nHeadBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, nHeadBuilder,
      bert.nHead, universe.typeOf[Int])
    bertBuilder.putAttr("nHead", nHeadBuilder.build)

    val intermediateSizeBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, intermediateSizeBuilder,
      bert.intermediateSize, universe.typeOf[Int])
    bertBuilder.putAttr("intermediateSize", intermediateSizeBuilder.build)

    val hiddenPDropBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, hiddenPDropBuilder,
      bert.hiddenPDrop, universe.typeOf[Double])
    bertBuilder.putAttr("hiddenPDrop", hiddenPDropBuilder.build)

    val attnPDropBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, attnPDropBuilder,
      bert.attnPDrop, universe.typeOf[Double])
    bertBuilder.putAttr("attnPDrop", attnPDropBuilder.build)

    val initializerRangeBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, initializerRangeBuilder,
      bert.initializerRange, universe.typeOf[Double])
    bertBuilder.putAttr("initializerRange", initializerRangeBuilder.build)

    val outputAllBlockBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, outputAllBlockBuilder,
      bert.outputAllBlock, universe.typeOf[Boolean])
    bertBuilder.putAttr("outputAllBlock", outputAllBlockBuilder.build)

    val embLabor = bert.embeddingLayer.labor.asInstanceOf[AbstractModule[Activity, Activity, T]]
    val subModule = ModuleSerializer.serialize(SerializeContext(ModuleData(embLabor,
      new ArrayBuffer[String](), new ArrayBuffer[String]()), context.storages,
      context.storageType, _copyWeightAndBias))
    bertBuilder.addSubModules(subModule.bigDLModule)

    val model = bert.
      asInstanceOf[KerasLayer[Activity, Activity, T]].labor
    val labor = model.asInstanceOf[KerasLayer[Activity, Activity, T]].labor
    val subModule2 = ModuleSerializer.serialize(SerializeContext(ModuleData(labor,
      new ArrayBuffer[String](), new ArrayBuffer[String]()), context.storages,
      context.storageType, _copyWeightAndBias))
    bertBuilder.addSubModules(subModule2.bigDLModule)

    val seqLenBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, seqLenBuilder,
      bert.seqLen, universe.typeOf[Int])
    bertBuilder.putAttr("seqLen", seqLenBuilder.build)

    val maxPositionLenBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, maxPositionLenBuilder,
      bert.maxPositionLen, universe.typeOf[Int])
    bertBuilder.putAttr("maxPositionLen", maxPositionLenBuilder.build)

    appendKerasLabel(context, bertBuilder)
  }
}
