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

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor._

import scala.math._
import com.intel.analytics.bigdl.numeric.NumericFloat
import scala.reflect.ClassTag

class SequenceBeamSearch[T: ClassTag] (
  val symbolsToLogitsFn: (Tensor[T], Tensor[Int], Map[String, Map[String, Tensor[T]]],
    Map[String, Tensor[T]]) =>
  (Tensor[T], Map[String, Map[String, Tensor[T]]], Map[String, Tensor[T]]),
  val vocabSize: Int,
  val batchSize: Int,
  val beamSize: Int,
  val alpha: Float,
  val maxDecodeLength: Int,
  val eosID: Int) (implicit ev: TensorNumeric[T]) {
  // val initialID = iniID
  // val initialCache = iniCa

  def expandDim(tensor: Tensor[T], axis: Int): Tensor[T] = {
    val shape = tensor.size()
    val newShape = shape.toBuffer
    newShape.insert(axis, 1)
    tensor.reshape(newShape.toArray)
  }

  // Tiles a given tensor by beam_size.
  def extendBeamSize(t: Tensor[T], beamSize: Int): Tensor[T] = {
    val tensor = expandDim(t, 1)
    val tileDim = new Array[Int](tensor.dim()).map(a => a + 1)
    tileDim(1) = beamSize
    tensor.repeatTensor(tileDim)
  }

  def lengthNormalization(alpha: Float, length: Int): T = {
    ev.pow(ev.fromType[Double](5.0 + length / 6.0), ev.fromType[Float](length))
  }

  def boolToFloat(b: Boolean): T = {
    if (b == true) ev.fromType[Double](1.0)
    else ev.fromType[Double](0.0)
  }

  def reduceAny(tensor: Tensor[Boolean]): Tensor[T] = {
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

  def reduceAll(tensor1: Tensor[T], tensor2: Tensor[T]): Boolean = {
    val sizeT = tensor1.size()
    var outputs = true
    for (i <- 1 to sizeT(0)) {
      outputs &&= ev.isGreater(tensor1.valueAt(i), tensor2.valueAt(i))
    }
    outputs
  }

  def continueSearch(state: Map[String, Any]): Boolean = {
    val inf = 1.0 * 1e7 * (-1)
    val i = state("CUR_INDEX").asInstanceOf[Tensor[Int]]
    val alive_log_probs = state("ALIVE_LOG_PROBS").asInstanceOf[Tensor[T]]
    var finished_scores = state("FINISHED_SCORES").asInstanceOf[Tensor[T]]
    val finished_flags = state("finishedFlags").asInstanceOf[Tensor[Boolean]]
    var notAtMaxDecodeLength = true
    if (i(Array(1)) < maxDecodeLength) {
      notAtMaxDecodeLength = true
    } else {
      notAtMaxDecodeLength = false
    }
    val maxLengthNorm = lengthNormalization(alpha, maxDecodeLength)
    val bestAliveScores = alive_log_probs.select(2, 1) / maxLengthNorm
    val newFinishedFlags = Tensor[T](finished_flags.size())
    newFinishedFlags.applyFun[Boolean](finished_flags, x => boolToFloat(x))
    finished_scores *= newFinishedFlags
    var lowestFinishedScores = finished_scores.min(2)._1
    lowestFinishedScores += (reduceAny(finished_flags) * ev.fromType[Double](-1.0)
      + ev.fromType[Double](1.0)) * ev.fromType[Double](inf)
      // (reduceAny(finished_flags) * (-1.0) + 1.0) * inf
    val worstFinishedScoreBetterThanBestAliveScore =
      reduceAll(lowestFinishedScores, bestAliveScores)
    notAtMaxDecodeLength && (!worstFinishedScoreBetterThanBestAliveScore)
  }

  // Reshapes first two dimensions in to single dimension.
  def flattenBeamDim(tensor: Tensor[T]): Tensor[T] = {
    val shape = tensor.size()
    val newShape = shape.toBuffer
    newShape(0) = shape(0) * shape(1)
    newShape.remove(1)
    tensor.reshape(newShape.toArray)
  }

  // Reshapes first dimension back to [batch_size, beam_size].
  def unFlattenBeamDim(tensor: Tensor[T], batchSize: Int, beamSize: Int): Tensor[T] = {
    val shape = tensor.size()
    val newShape = shape.toBuffer
    newShape(0) = batchSize
    newShape.insert(1, beamSize)
    tensor.reshape(newShape.toArray)
  }

  // logits - log(sum(exp(logits)))
  def logProbFromLogits(logits: Tensor[T]): Tensor[T] = {
    // val logits1 = logits.asInstanceOf[Tensor[Double]]
    val shape = logits.size()
    val getExp = Tensor[T](shape)
    getExp.applyFun[T](logits, x => ev.exp(x))
    val getSumExp = getExp.sum(3)
    val getLogSumExp = Tensor[T](getSumExp.size())
    getLogSumExp.applyFun[T](getSumExp, x => ev.log(x))
    logits - getLogSumExp.repeatTensor(Array(1, 1, shape(2)))
  }

  def gatherNd(tensor: Tensor[T], indices: Tensor[Int]): Tensor[T] = {
    // val indices1 = indices.asInstanceOf[Tensor[Int]]
    // val tensor1 = tensor.asInstanceOf[Tensor[Double]]
    val shape1 = tensor.size()
    val shape2 = indices.size()
    var slices = new Array[T](0)
    if (shape1.length > 2) {
      for (i <- 1 to shape2(0)) {
        for (j <- 1 to shape2(1)) {
          slices ++= tensor
            .select(2, indices.valueAt(i, j, 2))
            .select(1, indices.valueAt(i, j, 1)).toArray()
        }
      }
    } else {
      slices = new Array[T](shape2(0) * shape2(1))
      for (i <- 1 to shape2(0)) {
        for (j <- 1 to shape2(1)) {
          println((i - 1) * shape2(1) + j - 1)
          slices((i - 1) * shape2(1) + j - 1) = tensor
            .valueAt(indices.valueAt(i, j, 1), indices.valueAt(i, j, 2))
        }
      }
    }
    shape1(0) = shape2(0)
    shape1(1) = shape2(1)
    Tensor(slices, shape1)

  }

  def concat(tensor1: Tensor[T], tensor2: Tensor[T], dim: Int)(
    implicit ev: TensorNumeric[T]): Tensor[T] = {
    val shape1 = tensor1.size()
    val shape2 = tensor2.size()
    val array1 = tensor1.reshape(Array(shape1.product)).toArray()
    val array2 = tensor2.reshape(Array(shape2.product)).toArray()
    var outputsArray = new Array[T](0)
    var concatLength1 = 1
    var concatLength2 = 1
    for (i <- dim-1 until shape1.length) {
      concatLength1 *= shape1(i)
    }
    for (i <- dim-1 until shape2.length) {
      concatLength2 *= shape2(i)
    }
    val group1 = array1.grouped(concatLength1)
    val group2 = array2.grouped(concatLength2)
    while (group1.hasNext) {
      outputsArray ++= group1.next()
      outputsArray ++= group2.next()
    }
    val newShape = shape1
    newShape(dim-1) = shape1(dim-1) + shape2(dim-1)
    Tensor(outputsArray, newShape)
  }

  def gatherBeams(nested: Tensor[T], beamIndices: Tensor[Int],
                               batchSize: Int, newBeamSize: Int)
  : Tensor[T] = {
    // val beamIndices1 = beamIndices.asInstanceOf[Tensor[Int]]
    val batchPos = Tensor.range(0, batchSize*newBeamSize-1, 1)/ev.fromType[Int](newBeamSize)
    val newBatchPos = batchPos.apply1(e => ev.floor(e)).asInstanceOf[Tensor[Int]]
      // .reshape(Array(batchSize, newBeamSize))

    val coordinates = Tensor[Int](batchSize, newBeamSize, 2)
    for (i <- 1 to batchSize) {
      for (j <- 1 to newBeamSize) {
        coordinates.setValue(i, j, 1, newBatchPos.valueAt(i, j))
        coordinates.setValue(i, j, 2, beamIndices.valueAt(i, j))
      }
    }
    gatherNd(nested, coordinates)
  }

  def gatherTopkBeams(tensor: Tensor[T], scoreOrLogProb: Tensor[T],
    batchSize: Int, beamSize: Int): Tensor[_] = {
    val (_, topkIndexes) = scoreOrLogProb.topk(beamSize, -1, false)
    gatherBeams(tensor, topkIndexes.asInstanceOf[Tensor[Int]], batchSize, beamSize)
  }

  def growAliveSeq(state: Map[String, Any])(
  implicit ev: TensorNumeric[T]): (Tensor[T], Tensor[T],
    Map[String, Map[String, Tensor[T]]], Map[String, Tensor[T]]) = {
    val i = state("CUR_INDEX").asInstanceOf[Tensor[Int]]
    val aliveSeq = state("ALIVE_SEQ").asInstanceOf[Tensor[T]]
    val aliveLogProbs = state("ALIVE_LOG_PROBS").asInstanceOf[Tensor[T]]
    val aliveCache = state("ALIVE_CACHE").asInstanceOf[Map[String, Map[String, Tensor[T]]]]
    val aliveEncoderDecoder = state("ALIVE_CACHE_ENCODER_DECODER")
      .asInstanceOf[Map[String, Tensor[T]]]
    val beamsToKeep = 2 * beamSize
    val flatIds = flattenBeamDim(aliveSeq)
    var flatCache = aliveCache.mapValues(_.mapValues(flattenBeamDim(_)))
    var flatEncoderDecoderCache = aliveEncoderDecoder.mapValues(flattenBeamDim(_))
    val (flatLogits, flatCache1, flatEncoderDecoderCache1) =
      symbolsToLogitsFn(flatIds, i, flatCache, flatEncoderDecoderCache)
    val logits = unFlattenBeamDim(flatLogits, batchSize, beamSize)
    val newflatCache = flatCache1.mapValues(_.mapValues(unFlattenBeamDim(_, batchSize, beamSize)))
    val newflatEncoderDecoderCache = flatEncoderDecoderCache1.mapValues(
      unFlattenBeamDim(_, batchSize, beamSize))
    val candidateLogProbs = logProbFromLogits(logits)
    val logProbs = candidateLogProbs + expandDim(aliveLogProbs, 2)
    val flatLogProbs = logProbs.reshape(Array(logProbs.size().product
      /(beamSize*vocabSize), beamSize*vocabSize))
    val (topkLogProbs, topkIndices) = flatLogProbs.topk(beamsToKeep, -1, false)
    val newTopkIndices = topkIndices.asInstanceOf[Tensor[Int]]
    val topkBeamIndices = newTopkIndices/vocabSize
    var topkSeq = gatherBeams(aliveSeq, topkBeamIndices, batchSize, beamsToKeep)
    val flatBeamCache = newflatCache.mapValues(_.mapValues
    (gatherBeams(_, topkBeamIndices, batchSize, beamsToKeep)))
    val flatBeamEncoderDecoderCache = newflatEncoderDecoderCache.mapValues(
      gatherBeams(_, topkBeamIndices, batchSize, beamsToKeep))
    var topkIds = newTopkIndices.apply1(e => e%vocabSize).asInstanceOf[Tensor[T]]
    topkIds = expandDim(topkIds, 2)
    topkSeq = concat(topkSeq, topkIds, 3)
    (topkSeq, topkLogProbs, flatBeamCache, flatBeamEncoderDecoderCache)

  }

  def growNewAliveState(newSeq: Tensor[T], newLogProbs: Tensor[T],
    newCache: Map[String, Map[String, Tensor[T]]], newEncoderDecoderCache: Map[String, Tensor[T]])
    : Map[String, Any] = {
    val inf = 1.0 * 1e7 * (-1)
    val newFinishedFlag = Tensor[T](newLogProbs.size())
    val newSeqSelect = newSeq.select(3, newSeq.size()(2))
    newFinishedFlag.applyFun[T](newSeqSelect, x => boolToFloat(x == eosID))
    // var newLogProbs1 = newLogProbs.asInstanceOf[Tensor[Double]]
    val newLogProbs1 = newLogProbs + newFinishedFlag * ev.fromType[Double](inf)
    val topAliveSeq = gatherTopkBeams(newSeq, newLogProbs1, batchSize, beamSize)
    val topAliveLogProbs = gatherTopkBeams(newLogProbs1, newLogProbs1, batchSize, beamSize)
    val topNewCache = newCache.mapValues(_.mapValues
      (gatherTopkBeams(_, newLogProbs1, batchSize, beamSize)))
    val topNewEncoderDecoderCache = newEncoderDecoderCache
      .mapValues(gatherTopkBeams(_, newLogProbs1, batchSize, beamSize))
    Map("ALIVE_SEQ" -> topAliveSeq, "ALIVE_LOG_PROBS" -> topAliveLogProbs,
      "ALIVE_CACHE" -> topNewCache, "ALIVE_EncoderDecoderCACHE" -> topNewEncoderDecoderCache)
  }

  def getNewFinishedState(state: Map[String, Any], newSeq: Tensor[T], newLogProbs: Tensor[T])(
  implicit ev: TensorNumeric[T]): Map[String, Any] = {
    val inf = 1.0 * 1e7 * (-1)
    val i = state("CUR_INDEX").asInstanceOf[Tensor[Int]]
    var finishedSeq = state("FINISHED_SEQ").asInstanceOf[Tensor[T]]
    var finishedScores = state("FINISHED_SEQ").asInstanceOf[Tensor[T]]
    var finishedFlags = state("FINISHED_FLAGS").asInstanceOf[Tensor[T]]
    finishedSeq = concat(finishedSeq, Tensor[T](batchSize, beamSize, 1), 3)
    val lengthNorm = lengthNormalization(alpha, i.valueAt(1))
    var newScores = newLogProbs/lengthNorm
    val newFinishedFlag = Tensor[T](newScores.size())
    val finishedFlagsSelect = finishedFlags.select(3, newSeq.size()(2))
    newFinishedFlag.applyFun[T](finishedFlagsSelect, x => boolToFloat(x == eosID))
    newScores += (Tensor(newFinishedFlag.size()).fill(ev.fromType[Double](1.0)) - newFinishedFlag) *
      ev.fromType[Double](inf)
    finishedSeq = concat(finishedSeq, newSeq, 2)
    finishedScores = concat(finishedScores, newScores, 2)
    finishedFlags = concat(finishedFlags, newFinishedFlag, 2)
    val topFinishedSeq = gatherTopkBeams(finishedSeq, finishedScores, batchSize, beamSize)
    val topFinishedScores = gatherTopkBeams(finishedScores, finishedScores, batchSize, beamSize)
    val topFinishedFlags = gatherTopkBeams(finishedFlags, finishedScores, batchSize, beamSize)
    Map("FINISHED_SEQ"-> topFinishedSeq, "FINISHED_SCORES"->topFinishedScores,
      "FINISHED_FLAGS"->topFinishedFlags)
  }

  def searchStep[T: ClassTag](state: Map[String, Any]): Map[String, Any] = {
    val (newSeq, newLogProbs, newCache, newEncoderDecoderCache) = growAliveSeq(state)
    val aliveState = growNewAliveState(newSeq, newLogProbs, newCache, newEncoderDecoderCache)
    val finishedState = getNewFinishedState(state, newSeq, newLogProbs)
    var newState : Map[String, Any] = Map("CUR_INDEX" ->(state("CUR_INDEX")
      .asInstanceOf[Tensor[Int]] + 1)) ++ aliveState ++ aliveState
    newState
  }

  def createInitialState(initialID: Tensor[T], initialCache:
  Map[String, Map[String, Tensor[T]]], initialEncoderDecoder: Map[String, Tensor[T]]):
  (Map[String, Any], Map[String, Any]) = {
    var curIndex = Tensor(Array(0), Array(1))
    var aliveSeq = extendBeamSize(initialID, beamSize)
    aliveSeq = expandDim(aliveSeq, 2)
    val inf = 1.0 * 1e7 * (-1)
    var initialLogProbs = Tensor[Double](beamSize).apply1(e => inf)
    initialLogProbs.setValue(1, 0.0)
    val aliveLogProbs = initialLogProbs.repeatTensor(Array(batchSize, 1))
    val aliveCache = initialCache.mapValues(_.mapValues(extendBeamSize(_, beamSize)))
    val aliveEncoderDecoder = initialEncoderDecoder.mapValues(extendBeamSize(_, beamSize))
    val finishedSeq = Tensor[Int](aliveSeq.size())
    val finishedScores = Tensor.ones[Double](batchSize, beamSize) * inf
    val finishedFlags = Tensor[Boolean](batchSize, beamSize)

    val state = Map("CUR_INDEX" -> curIndex,
      "ALIVE_SEQ" -> aliveSeq,
      "ALIVE_LOG_PROBS" -> aliveLogProbs,
      "ALIVE_CACHE" -> aliveCache,
      "ALIVE_CACHE_ENCODER_DECODER" -> aliveEncoderDecoder,
      "FINISHED_SEQ" -> finishedSeq,
      "FINISHED_SCORES" -> finishedScores,
      "FINISHED_FLAGS" -> finishedFlags)

    val stateShapeInvariants = Map("CUR_INDEX" -> Tensor(),
      "ALIVE_SEQ" -> Tensor(null, beamSize, null),
      "ALIVE_LOG_PROBS" -> Tensor(null, beamSize, null),
      "ALIVE_CACHE" -> aliveCache,
      "ALIVE_CACHE_ENCODER_DECODER" -> aliveEncoderDecoder,
      "FINISHED_SEQ" -> finishedSeq,
      "FINISHED_SCORES" -> finishedScores,
      "finishedFlags" -> finishedFlags)

    return (state, stateShapeInvariants)
  }

  def where(tensor: Tensor[T], a: Tensor[T], b: Tensor[T])
  (implicit ev: TensorNumeric[T]): Tensor[T] = {
    val arrayBool = tensor.toArray()
    val arrayA = a.toArray()
    val arrayB = b.toArray()
    for (i <- arrayBool.indices) {
      if (arrayBool(i) == 0) arrayA(i) = arrayB(i)
    }
    Tensor(arrayA, tensor.size())
  }
  def search(initialID: Tensor[T], initialCache:
  Map[String, Map[String, Tensor[T]]], initialEncoderDecoder: Map[String, Tensor[T]])
  (implicit ev: TensorNumeric[T])
  : (Tensor[T], Tensor[T]) = {
    var (state, stateShape) = createInitialState(initialID, initialCache, initialEncoderDecoder)
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
    (finishedSeq, finishedScores)
  }
}

object SequenceBeamSearch{
  def apply[@specialized(Float, Double) T: ClassTag](
    symbolsToLogitsFn: (Tensor[T], Tensor[Int], Map[String, Map[String, Tensor[T]]],
    Map[String, Tensor[T]]) =>
    (Tensor[T], Map[String, Map[String, Tensor[T]]], Map[String, Tensor[T]]),
    vocabSize: Int,
    batchSize: Int,
    beamSize: Int,
    alpha: Float,
    maxDecodeLength: Int,
    eosID: Int)
    (implicit ev: TensorNumeric[T]) : SequenceBeamSearch[T] = {
    new SequenceBeamSearch[T](
      symbolsToLogitsFn,
      vocabSize,
      batchSize,
      beamSize,
      alpha,
      maxDecodeLength,
      eosID)
  }

}