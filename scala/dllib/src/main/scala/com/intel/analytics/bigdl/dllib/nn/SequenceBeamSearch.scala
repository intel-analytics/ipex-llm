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
package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * Beam search to find the translated sequence with the highest probability.
 * @param vocabSize size of tokens
 * @param beamSize number of beams
 * @param alpha defining the strength of length normalization
 * @param maxDecodeLength maximum length to decoded sequence
 * @param eosID id of eos token, used to determine when a sequence has finished
 * @param numHiddenLayers number of hidden layers
 * @param hiddenSize size of hidden layer
 */
class SequenceBeamSearch[T: ClassTag](
  val vocabSize: Int,
  val beamSize: Int,
  val alpha: Float,
  val maxDecodeLength: Int,
  val eosID: Float,
  val numHiddenLayers: Int,
  val hiddenSize: Int)(implicit ev: TensorNumeric[T])
  extends AbstractModule[Table, Activity, T] {

  private val inf = 1e7f * (-1)
  private var batchSize = 0
  private val newFinishedFlags = Tensor[T]
  private var aliveLogProbs = Tensor[T]
  private var finishedSeq = Tensor[T]
  private var aliveSeq = Tensor[T]
  private var finishedFlags = Tensor[Boolean]
  private val finishedFlagsSeq = Tensor[T]
  private var finishedScores = Tensor[T]
  private val gatherTensor = Tensor[T]
  private val topkSeq = Tensor[T]
  private val topkLogProbs = Tensor[T]
  private val topkScore = Tensor[T]
  private val topkFlags = Tensor[T]
  private var symbolToLogits: (Tensor[T], Int, Int, Tensor[T], Tensor[T], List[Tensor[T]],
    List[Tensor[T]]) => (Tensor[T], Tensor[T], Tensor[T], List[Tensor[T]], List[Tensor[T]]) = null
  private val topkEncoder = Tensor[T]
  private val topkAttentionBias = Tensor[T]
  private var topkLayerK: List[Tensor[T]] = List()
  private var topkLayerV: List[Tensor[T]] = List()
  for (i <- 1 to  numHiddenLayers) {
    val tensor1 = Tensor[T]
    val tensor2 = Tensor[T]
    topkLayerK ++= List(tensor1)
    topkLayerV ++= List(tensor2)
  }

  private def expandDim(tensor: Tensor[T], axis: Int): Tensor[T] = {
    val shape = tensor.size()
    val newShape = shape.toBuffer
    newShape.insert(axis, 1)
    tensor.reshape(newShape.toArray)
  }

  // Tiles a given tensor by beam_size.
  private def extendBeamSize(t: Tensor[T], beamSize: Int): Tensor[T] = {
    val tensor = expandDim(t, 1)
    val tileDim = new Array[Int](tensor.dim()).map(a => a + 1)
    tileDim(1) = beamSize
    tensor.repeatTensor(tileDim)
  }

  private def lengthNormalization(alpha: Float, length: Int): T = {
    ev.pow(ev.fromType[Double](5.0 + length / 6.0), ev.fromType[Float](alpha))
  }

  private def boolToFloat(b: Boolean): T = {
    if (b) ev.one
    else ev.zero
  }

  private def floatToBool(f: T): Boolean = {
    if (f == 1.0) true
    else false
  }

  // Computes the "logical or" of elements across dimensions of a tensor.
  private def reduceAny(tensor: Tensor[Boolean]): Tensor[T] = {
    val tSize = tensor.size()
    val outputs = Tensor[T](tSize(0))
    for (i <- 1 to tSize(0)) {
      var valueAny = false
      for (j <- 1 to tSize(1)) {
        valueAny = valueAny || tensor.valueAt(i, j)
      }
      outputs.setValue(i, boolToFloat(valueAny))
    }
    outputs
  }

  // Computes the "logical and" of elements across dimensions of a tensor.
  private def reduceAll(tensor1: Tensor[T], tensor2: Tensor[T]): Boolean = {
    val sizeT = tensor1.size()
    var outputs = true
    for (i <- 1 to sizeT(0)) {
      outputs &&= ev.isGreater(tensor1.valueAt(i), tensor2.valueAt(i))
    }
    outputs
  }

  /**
   * Return whether to continue the search loop.
   * The loops should terminate when
   * 1) when decode length has been reached, or
   * 2) when the worst score in the finished sequences is better than the best
   * score in the alive sequences (i.e. the finished sequences are provably
   * unchanging)
   *
   * @param state A map with the current loop state.
   * @return Boolean value with value True if loop should continue, False if loop should
   *         terminate.
   */
  private def continueSearch(state: Map[String, Any]): Boolean = {
    val i = state("CUR_INDEX").asInstanceOf[Int]
    finishedFlags = state("FINISHED_FLAGS").asInstanceOf[Tensor[Boolean]]
    aliveLogProbs.copy(state("ALIVE_LOG_PROBS").asInstanceOf[Tensor[T]])
    finishedScores.resizeAs(state("FINISHED_SCORES").asInstanceOf[Tensor[T]])
      .copy(state("FINISHED_SCORES").asInstanceOf[Tensor[T]])
    var notAtMaxDecodeLength = true
    if (i < maxDecodeLength) {
      notAtMaxDecodeLength = true
    } else {
      notAtMaxDecodeLength = false
    }
    val maxLengthNorm = lengthNormalization(alpha, maxDecodeLength)
    // Get the best possible scores from alive sequences.
    val bestAliveScores = aliveLogProbs.select(2, 1) / maxLengthNorm
    newFinishedFlags.applyFun[Boolean](finishedFlags, x => boolToFloat(x))
    finishedScores.cmul(newFinishedFlags)
    // Compute worst score in finished sequences for each batch element
    var lowestFinishedScores = finishedScores.min(2)._1
    lowestFinishedScores += (reduceAny(finishedFlags) * ev.fromType[Double](-1.0)
      + ev.fromType[Double](1.0)) * ev.fromType[Double](inf)
    lowestFinishedScores = lowestFinishedScores.reshape(Array(lowestFinishedScores.size()(0)))
    val worstFinishedScoreBetterThanBestAliveScore =
      reduceAll(lowestFinishedScores, bestAliveScores)
    notAtMaxDecodeLength && (!worstFinishedScoreBetterThanBestAliveScore)
  }

  // Reshapes first two dimensions in to single dimension.
  private def flattenBeamDim(tensor: Tensor[T]): Tensor[T] = {
    val shape = tensor.size()
    val newShape = shape.toBuffer
    newShape(0) = shape(0) * shape(1)
    newShape.remove(1)
    tensor.reshape(newShape.toArray)
  }

  // Reshapes first dimension back to [batch_size, beam_size].
  private def unFlattenBeamDim(tensor: Tensor[T], batchSize: Int, beamSize: Int): Tensor[T] = {
    val shape = tensor.size()
    val newShape = shape.toBuffer
    newShape(0) = batchSize
    newShape.insert(1, beamSize)
    tensor.reshape(newShape.toArray)
  }

  // logits - log(sum(exp(logits)))
  private def logProbFromLogits(logits: Tensor[T]): Tensor[T] = {
    val shape = logits.size()
    val getExp = Tensor[T](shape)
    getExp.applyFun[T](logits, x => ev.exp(x))
    val getSumExp = getExp.sum(3)
    val getLogSumExp = Tensor[T](getSumExp.size())
    getLogSumExp.applyFun[T](getSumExp, x => ev.log(x))
    logits - getLogSumExp.repeatTensor(Array(1, 1, shape(2)))
  }

  // Gather slices from tensor into outputs with shape specified by indices.
  private def gatherNd(tensor: Tensor[T], indices: Tensor[T], outputs: Tensor[T]): Tensor[T] = {
    val shape1 = tensor.size()
    val shape2 = indices.size()
    var slices = new Array[T](0)
    if (shape1.length == 2) {
      outputs.resize(shape2(0), shape2(1))
      slices = new Array[T](shape2(0) * shape2(1))
      for (i <- 1 to shape2(0)) {
        for (j <- 1 to shape2(1)) {
          slices((i - 1) * shape2(1) + j - 1) = tensor.valueAt(ev.toType[Int](ev.plus
          (indices.valueAt(i, j, 1), ev.fromType[Float](1.0f))), ev.toType[Int]
            (ev.plus(indices.valueAt(i, j, 2), ev.fromType[Float](1.0f))))
        }
      }
    } else if (shape1.length == 3) {
      outputs.resize(shape2(0), shape2(1), shape1(2))
      for (i <- 1 to shape2(0)) {
        for (j <- 1 to shape2(1)) {
          slices ++= tensor
            .select(2, ev.toType[Int](ev.plus(indices.valueAt(i, j, 2), ev.fromType[Float](1.0f))))
            .select(1, ev.toType[Int](ev.plus(indices.valueAt(i, j, 1), ev.fromType[Float](1.0f))))
            .toArray()
        }
      }
    } else if (shape1.length == 4) {
      outputs.resize(shape2(0), shape2(1), shape1(2), shape1(3))
      for (i <- 1 to shape2(0)) {
        for (j <- 1 to shape2(1)) {
          slices ++= tensor
            .select(2, ev.toType[Int](ev.plus(indices.valueAt(i, j, 2), ev.fromType[Float](1.0f))))
            .select(1, ev.toType[Int](ev.plus(indices.valueAt(i, j, 1), ev.fromType[Float](1.0f))))
            .reshape(Array(shape1(2) * shape1(3)))toArray()
        }
      }
    } else if (shape1.length == 5) {
      outputs.resize(shape2(0), shape2(1), shape1(2), shape1(3), shape1(4))
      for (i <- 1 to shape2(0)) {
        for (j <- 1 to shape2(1)) {
          slices ++= tensor
            .select(2, ev.toType[Int](ev.plus(indices.valueAt(i, j, 2), ev.fromType[Float](1.0f))))
            .select(1, ev.toType[Int](ev.plus(indices.valueAt(i, j, 1), ev.fromType[Float](1.0f))))
            .reshape(Array(shape1(2) * shape1(3) * shape1(4)))toArray()
        }
      }
    }
    val outputData = outputs.storage().array()
    val outputOffset = outputs.storageOffset() - 1
    for(i <- slices.indices) {
      outputData(outputOffset + i) = slices(i)
    }
    shape1(0) = shape2(0)
    shape1(1) = shape2(1)
    outputs
  }

  // Concatenates tensor1 and tensor2 along one dimension.
  private def concat(tensor1: Tensor[T], tensor2: Tensor[T], dim: Int): Tensor[T] = {
    val shape1 = tensor1.size()
    val shape2 = tensor2.size()
    val array1 = tensor1.reshape(Array(shape1.product)).toArray()
    val array2 = tensor2.reshape(Array(shape2.product)).toArray()
    var outputsArray = new Array[T](0)
    var concatLength1 = 1
    var concatLength2 = 1
    for (i <- dim - 1 until shape1.length) {
      concatLength1 *= shape1(i)
    }
    for (i <- dim - 1 until shape2.length) {
      concatLength2 *= shape2(i)
    }
    val group1 = array1.grouped(concatLength1)
    val group2 = array2.grouped(concatLength2)
    while (group1.hasNext) {

      outputsArray ++= group1.next()
      outputsArray ++= group2.next()
    }
    val newShape = shape1
    newShape(dim - 1) = shape1(dim - 1) + shape2(dim - 1)
    Tensor(outputsArray, newShape)
  }

  // Gather beams from tensors.
  private def gatherBeams(nested: Tensor[T], beamIndices: Tensor[T],
    batchSize: Int, newBeamSize: Int): Tensor[T] = {
    val batchPos = (Tensor.range(0, batchSize * newBeamSize - 1, 1) / ev.fromType[Int](newBeamSize))
      .reshape(Array(batchSize, newBeamSize))
    val newBatchPos = batchPos.apply1(e => ev.floor(e))
    val coordinates = Tensor[T](batchSize, newBeamSize, 2)
    for (i <- 1 to batchSize) {
      for (j <- 1 to newBeamSize) {
        coordinates.setValue(i, j, 1, newBatchPos.valueAt(i, j))
        coordinates.setValue(i, j, 2, beamIndices.valueAt(i, j))
      }
    }
    gatherNd(nested, coordinates.asInstanceOf[Tensor[T]], gatherTensor)
  }

  // Gather top beams from nested structure.
  private def gatherTopkBeams(tensor: Tensor[T], scoreOrLogProb: Tensor[T],
    batchSize: Int, beamSize: Int): Tensor[T] = {
    val (_, topkIndexes) = scoreOrLogProb.topk(beamSize, -1, false)
    topkIndexes.apply1(e => ev.minus(e, ev.fromType[Float](1.0f)))
    gatherBeams(tensor, topkIndexes, batchSize, beamSize)
  }

  def setLogitFn(fn: (Tensor[T], Int, Int, Tensor[T], Tensor[T], List[Tensor[T]],
    List[Tensor[T]]) => (Tensor[T], Tensor[T], Tensor[T], List[Tensor[T]], List[Tensor[T]])):
    SequenceBeamSearch[T] = {
    symbolToLogits = fn
    this
  }

  /**
   * Grow alive sequences by one token, and collect top 2*beam_size sequences.
   * 2*beam_size sequences are collected because some sequences may have reached
   * the EOS token. 2*beam_size ensures that at least beam_size sequences are
   * still alive.
   * @param state A map with the current loop state.
   * @return newSeq Top 2*beam_size sequences [batch_size, 2 * beam_size, cur_index + 1]
   *         topkLogProbs probabilities of returned sequences [batch_size, 2 * beam_size]
   */
  private def growAliveSeq(state: Map[String, Any]): (Tensor[T], Tensor[T]) = {
    val i = state("CUR_INDEX").asInstanceOf[Int]
    aliveSeq = state("ALIVE_SEQ").asInstanceOf[Tensor[T]]
    aliveLogProbs = state("ALIVE_LOG_PROBS").asInstanceOf[Tensor[T]]
    val aliveEncoder = state("ENCODER").asInstanceOf[Tensor[T]]
    val aliveAttentionsBias = state("ATTENTION_BIAS").asInstanceOf[Tensor[T]]
    val aliveLayerK = state("LAYERK").asInstanceOf[List[Tensor[T]]]
    val aliveLayerV = state("LAYERV").asInstanceOf[List[Tensor[T]]]
    val beamsToKeep = 2 * beamSize
    val flatIds = flattenBeamDim(aliveSeq)
    val flatEncoder = flattenBeamDim(aliveEncoder)
    val flatAttentionBias = flattenBeamDim(aliveAttentionsBias)
    val flatLayerK = aliveLayerK.map(e => flattenBeamDim(e))
    val flatLayerV = aliveLayerV.map(e => flattenBeamDim(e))
    var (flatLogits, newFlatEncoder, newAttentionBias, newFlatLayerK,
     newFlatLayerV) = symbolToLogits(flatIds, i, maxDecodeLength, flatEncoder,
     flatAttentionBias, flatLayerK, flatLayerV)
    newFlatEncoder = unFlattenBeamDim(newFlatEncoder, batchSize, beamSize)
    newAttentionBias = unFlattenBeamDim(newAttentionBias, batchSize, beamSize)
    newFlatLayerK = newFlatLayerK.map(e => unFlattenBeamDim(e, batchSize, beamSize))
    newFlatLayerV = newFlatLayerV.map(e => unFlattenBeamDim(e, batchSize, beamSize))
    val logits = unFlattenBeamDim(flatLogits, batchSize, beamSize)
    val candidateLogProbs = logProbFromLogits(logits)
    val logProbs = candidateLogProbs + expandDim(aliveLogProbs, 2)
      .repeatTensor(Array(1, 1, vocabSize))
    val flatLogProbs = logProbs.reshape(Array(logProbs.size().product
      / (beamSize * vocabSize), beamSize * vocabSize))
    val (topkLogProbs, topkIndices) = flatLogProbs.topk(beamsToKeep, -1, false)
    topkIndices.apply1(e => ev.minus(e, ev.fromType[Float](1.0f)))
    val topkBeamIndices = (topkIndices / ev.fromType[Int](vocabSize)).apply1(e => ev.floor(e))
    // Extract the alive sequences that generate the highest log probabilities
    var gatherTmp = gatherBeams(aliveSeq, topkBeamIndices, batchSize, beamsToKeep)
    topkSeq.resizeAs(gatherTmp).copy(gatherTmp)
    gatherTmp = gatherBeams(newFlatEncoder, topkBeamIndices, batchSize, beamsToKeep)
    topkEncoder.resizeAs(gatherTmp).copy(gatherTmp)
    gatherTmp = gatherBeams(newAttentionBias, topkBeamIndices, batchSize, beamsToKeep)
    topkAttentionBias.resizeAs(gatherTmp).copy(gatherTmp)
    for (i <- 0 until numHiddenLayers) {
      gatherTmp = gatherBeams(newFlatLayerK(i), topkBeamIndices, batchSize, beamsToKeep)
      topkLayerK(i).resizeAs(gatherTmp).copy(gatherTmp)
      gatherTmp = gatherBeams(newFlatLayerV(i), topkBeamIndices, batchSize, beamsToKeep)
      topkLayerV(i).resizeAs(gatherTmp).copy(gatherTmp)
    }
    var topkIds = topkIndices.apply1(e => ev.fromType[Int](ev.toType[Int](e) % vocabSize))
    topkIds = expandDim(topkIds, 2)
    val newSeq = concat(topkSeq, topkIds, 3)
    (newSeq, topkLogProbs)
  }

  /**
   * Gather the top k sequences that are still alive.
   * @param newSeq New sequences generated by growing the current alive sequences
   * @param newLogProbs Log probabilities of new sequences
   * @return map with alive keys
   */
  private def growNewAliveState(newSeq: Tensor[T], newLogProbs: Tensor[T]): Map[String, Any] = {
    finishedFlagsSeq.copy(newSeq.select(3, newSeq.size()(2)))
    finishedFlagsSeq.apply1(x => boolToFloat(ev.toType[Float](x) == eosID))
    val newLogProbs1 = newLogProbs + finishedFlagsSeq * ev.fromType[Double](inf)
    var gatherTmp = gatherTopkBeams(newSeq, newLogProbs1, batchSize, beamSize)
    aliveSeq.resizeAs(gatherTmp).copy(gatherTmp)
    gatherTmp = gatherTopkBeams(newLogProbs1, newLogProbs1, batchSize, beamSize)
    topkLogProbs.resizeAs(gatherTmp).copy(gatherTmp)
    gatherTmp = gatherTopkBeams(topkEncoder, newLogProbs1, batchSize, beamSize)
    topkEncoder.resizeAs(gatherTmp).copy(gatherTmp)
    gatherTmp = gatherTopkBeams(topkAttentionBias, newLogProbs1, batchSize, beamSize)
    topkAttentionBias.resizeAs(gatherTmp).copy(gatherTmp)
    for (i <- 0 until numHiddenLayers) {
      gatherTmp = gatherTopkBeams(topkLayerK(i), newLogProbs1, batchSize, beamSize)
      topkLayerK(i).resizeAs(gatherTmp).copy(gatherTmp)
      gatherTmp = gatherTopkBeams(topkLayerV(i), newLogProbs1, batchSize, beamSize)
      topkLayerV(i).resizeAs(gatherTmp).copy(gatherTmp)
    }
    Map("ALIVE_SEQ" -> aliveSeq, "ALIVE_LOG_PROBS" -> topkLogProbs,
      "ENCODER" -> topkEncoder, "ATTENTION_BIAS" -> topkAttentionBias,
      "LAYERK" -> topkLayerK, "LAYERV" -> topkLayerV)
  }

  /**
   * Combine new and old finished sequences, and gather the top k sequences.
   * @param state A map with the current loop state.
   * @param newSeq New sequences generated by growing the current alive sequences
   * @param newLogProbs Log probabilities of new sequences
   * @return map with finished keys
   */
  private def getNewFinishedState(state: Map[String, Any], newSeq: Tensor[T],
    newLogProbs: Tensor[T]): Map[String, Any] = {
    val i = state("CUR_INDEX").asInstanceOf[Int]
    finishedSeq = state("FINISHED_SEQ").asInstanceOf[Tensor[T]]
    finishedScores = state("FINISHED_SCORES").asInstanceOf[Tensor[T]]
    finishedFlags = state("FINISHED_FLAGS").asInstanceOf[Tensor[Boolean]]
    // append a column of 0-ids to finished_seq to increment the length.
    finishedSeq = concat(finishedSeq, Tensor[T](batchSize, beamSize, 1), 3)
    val lengthNorm = lengthNormalization(alpha, i)
    var newScores = newLogProbs / lengthNorm
    // Set the scores of the still-alive seq in new_seq to large negative values.
    newScores += (Tensor(finishedFlagsSeq.size()).fill(ev.fromType[Float](1.0f))
     - finishedFlagsSeq) * ev.fromType[Float](inf)
    // Combine sequences, scores, and flags.
    finishedSeq = concat(finishedSeq, newSeq, 2)
    finishedScores = concat(finishedScores, newScores, 2)
    var finishedFlags1 = Tensor[T](finishedFlags.size())
    finishedFlags1.applyFun[Boolean](finishedFlags, x => boolToFloat(x))
    finishedFlags1 = concat(finishedFlags1, finishedFlagsSeq, 2)
    var gatherTmp = gatherTopkBeams(finishedSeq, finishedScores, batchSize, beamSize)
    topkSeq.resizeAs(gatherTmp).copy(gatherTmp)
    gatherTmp = gatherTopkBeams(finishedScores, finishedScores, batchSize, beamSize)
    topkScore.resizeAs(gatherTmp).copy(gatherTmp)
    gatherTmp = gatherTopkBeams(finishedFlags1, finishedScores, batchSize, beamSize)
    topkFlags.resizeAs(gatherTmp).copy(gatherTmp)
    val topFinishedFlags1 = topkFlags.reshape(Array(topkFlags.size().product))
      .toArray()
    val outputFlag = ArrayBuffer[Boolean]()
    for (ele <- topFinishedFlags1) {
      outputFlag.append(floatToBool(ele))
    }
    finishedFlags = Tensor(outputFlag.toArray, topkFlags.size())
    finishedSeq.resizeAs(topkSeq).copy(topkSeq)
    Map("FINISHED_SEQ" -> finishedSeq, "FINISHED_SCORES" -> topkScore,
      "FINISHED_FLAGS" -> finishedFlags)
  }

  /**
   * Grow alive sequences by a single ID. Sequences that have reached the EOS
   * token are marked as finished. The alive and finished sequences with the
   * highest log probabilities and scores are returned.
   */
  private def searchStep(state: Map[String, Any]): Map[String, Any] = {
    val (newSeq, newLogProbs) = growAliveSeq(state)
    val aliveState = growNewAliveState(newSeq, newLogProbs)
    val finishedState = getNewFinishedState(state, newSeq, newLogProbs)
    val newState: Map[String, Any] = Map("CUR_INDEX" -> (state("CUR_INDEX")
      .asInstanceOf[Int] + 1)) ++ aliveState ++ finishedState
    newState
  }

  // return initial state map
  private def createInitialState(encoderOutputs: Tensor[T], encoderDecoderAttentionBias: Tensor[T]):
    Map[String, Any] = {
    batchSize = encoderOutputs.size()(0)
    newFinishedFlags.resize(batchSize, beamSize)
    aliveLogProbs.resize(batchSize, beamSize)
    finishedFlags.resize(batchSize, beamSize)
    finishedFlagsSeq.resize(batchSize, beamSize * 2)
    finishedScores.resize(batchSize, beamSize)
    val curIndex = 0
    val initialID = Tensor[T](Array(batchSize))
    var initialAliveSeq = extendBeamSize(initialID, beamSize)
    initialAliveSeq = expandDim(initialAliveSeq, 2)
    var initialLogProbs = Tensor[T](beamSize).apply1(e => ev.fromType[Float](inf))
    initialLogProbs.setValue(1, ev.fromType[Float](0.0f))
    initialLogProbs = initialLogProbs.repeatTensor(Array(batchSize, 1))
    val aliveEncoder = extendBeamSize(encoderOutputs, beamSize)
    val aliveAttentionsBias = extendBeamSize(encoderDecoderAttentionBias, beamSize)
    var aliveLayerK: List[Tensor[T]] = List()
    var aliveLayerV: List[Tensor[T]] = List()
    for (i <- 1 to  numHiddenLayers) {
      val tensor1 = Tensor[T](batchSize, beamSize, 0, hiddenSize)
      val tensor2 = Tensor[T](batchSize, beamSize, 0, hiddenSize)
      aliveLayerK ++= List(tensor1)
      aliveLayerV ++= List(tensor2)
    }
    val initialFinishedSeq = Tensor[T](initialAliveSeq.size())
    val initialFinishedScores = Tensor.ones[T](batchSize, beamSize) * ev.fromType[Float](inf)
    val initialFinishedFlags = Tensor[Boolean](batchSize, beamSize)

    val state = Map("CUR_INDEX" -> curIndex,
      "ALIVE_SEQ" -> initialAliveSeq,
      "ALIVE_LOG_PROBS" -> initialLogProbs,
      "ENCODER" -> aliveEncoder,
      "ATTENTION_BIAS" -> aliveAttentionsBias,
      "LAYERK" -> aliveLayerK,
      "LAYERV" -> aliveLayerV,
      "FINISHED_SEQ" -> initialFinishedSeq,
      "FINISHED_SCORES" -> initialFinishedScores,
      "FINISHED_FLAGS" -> initialFinishedFlags)
    state
  }

  // replace value in a with b according to tensor value
  private def where(tensor: Tensor[T], a: Tensor[T], b: Tensor[T]): Tensor[T] = {
    val arrayBool = tensor.toArray()
    val shape = a.size()
    for (i <- arrayBool.indices) {
      if (arrayBool(i) == 0) {
        if (shape.length == 3) {
          for (j <- 1 to shape(1)) {
            for (k <- 1 to shape(2)) {
              a.setValue(i + 1, j, k, b.valueAt(i + 1, j, k))
            }
          }
        } else {
          for (j <- 1 to shape(1)) {
            a.setValue(i + 1, j, b.valueAt(i + 1, j))
          }
        }
      }
    }
    a
  }

  override def updateOutput(input: Table): Activity = {
    val encoderOutputs = input[Tensor[T]](1)
    val encoderDecoderAttentionBias = input[Tensor[T]](2)
    require(symbolToLogits != null, "symbolToLogits function is null, please set this function")
    var state = createInitialState(encoderOutputs, encoderDecoderAttentionBias)
    while (continueSearch(state)) {
      state = searchStep(state)
    }
    val finishedState = state
    val aliveSeq = finishedState("ALIVE_SEQ").asInstanceOf[Tensor[T]]
    val aliveLogProbs = finishedState("ALIVE_LOG_PROBS").asInstanceOf[Tensor[T]]
    var finishedSeq = finishedState("FINISHED_SEQ").asInstanceOf[Tensor[T]]
    var finishedScores = finishedState("FINISHED_SCORES").asInstanceOf[Tensor[T]]
    val finishedFlags = finishedState("FINISHED_FLAGS").asInstanceOf[Tensor[Boolean]]
    finishedSeq = where(reduceAny(finishedFlags), finishedSeq, aliveSeq)
    finishedScores = where(reduceAny(finishedFlags), finishedScores, aliveLogProbs)
    output = T(finishedSeq, finishedScores)
    output
  }

  override def updateGradInput(input: Table, gradOutput: Activity): Table = {
    gradInput = gradOutput.toTable
    gradInput
  }

  override def clearState(): this.type = {
    super.clearState()
    batchSize = 0
    newFinishedFlags.set()
    aliveLogProbs.set()
    finishedSeq.set()
    aliveSeq.set()
    finishedFlags.set()
    finishedFlagsSeq.set()
    finishedScores.set()
    gatherTensor.set()
    topkSeq.set()
    topkLogProbs.set()
    topkScore.set()
    topkFlags.set()
    topkEncoder.set()
    topkAttentionBias.set()
    topkLayerK.foreach(e => e.set())
    topkLayerV.foreach(e => e.set())
    this
  }
}

object SequenceBeamSearch {
  def apply[@specialized(Float, Double) T: ClassTag](
    vocabSize: Int,
    beamSize: Int,
    alpha: Float,
    maxDecodeLength: Int,
    eosID: Float,
    numHiddenLayers: Int,
    hiddenSize: Int)
  (implicit ev: TensorNumeric[T]): SequenceBeamSearch[T] = {
    new SequenceBeamSearch[T](
      vocabSize,
      beamSize,
      alpha,
      maxDecodeLength,
      eosID,
      numHiddenLayers,
      hiddenSize)
  }
}
