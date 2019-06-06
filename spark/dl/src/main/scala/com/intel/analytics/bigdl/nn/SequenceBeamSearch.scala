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

import com.intel.analytics.bigdl.tensor._
import scala.math._

import scala.reflect.ClassTag

class SequenceBeamSearch[T: ClassTag]
(val fn: (Tensor[T], Tensor[Int], Map[String, Map[String, Tensor[T]]], Map[String, Tensor[T]]) =>
  (Tensor[T], Map[String, Map[String, Tensor[T]]], Map[String, Tensor[T]]),
 val vo: Int, val bas: Int, val bes: Int, val al: Float, val maxDecode: Int, val id: Int) {
  val symbolsToLogitsFn = fn
  val vocabSize = vo
  val batchSize = bas
  val beamSize = bes
  val alpha = al
  val maxDecodeLength = maxDecode
  val eosID = id
  // val initialID = iniID
  // val initialCache = iniCa

  def expandDim[T: ClassTag](tensor: Tensor[T], axis: Int): Tensor[T] = {
    val shape = tensor.size()
    val newShape = shape.toBuffer
    newShape.insert(axis, 1)
    return tensor.reshape(newShape.toArray)
  }

  // Tiles a given tensor by beam_size.
  def extendBeamSize(t: Tensor[T], beamSize: Int): Tensor[T] = {
    val tensor = expandDim(t, 1)
    val tileDim = new Array[Int](tensor.dim()).map(a => a + 1)
    tileDim(1) = beamSize
    return tensor.repeatTensor(tileDim)
  }

  def lengthNormalization(alpha: Float, length: Int): Double = {
    return pow((5.0 + length / 6.0), length)
  }

  def boolToFloat(b: Boolean): Double = {
    if (b == true) 1.0
    else 0.0
  }

  def reduceAny(tensor: Tensor[Boolean]): Tensor[Double] = {
    val tSize = tensor.size()
    val outputs = Tensor[Double](tSize(0))
    for (i <- 1 to tSize(0)) {
      var valueAny = false
      for (j <- 1 to tSize(1)) {
        valueAny = valueAny || tensor.valueAt(i, j)
      }
      outputs.setValue(i, boolToFloat(valueAny))
    }
    outputs
  }

  def reduceAll(tensor1: Tensor[Double], tensor2: Tensor[Double]): Boolean = {
    val sizeT = tensor1.size()
    var outputs = true
    for (i <- 1 to sizeT(0)) {
      outputs &&= (tensor1.valueAt(i) > tensor2.valueAt(i))
    }
    outputs
  }

  def continueSearch(state: Map[String, Any]): Boolean = {
    val inf = 1.0 * 1e7 * (-1)
    val i = state("CUR_INDEX").asInstanceOf[Tensor[Int]]
    val alive_log_probs = state("ALIVE_LOG_PROBS").asInstanceOf[Tensor[Double]]
    var finished_scores = state("FINISHED_SCORES").asInstanceOf[Tensor[Double]]
    val finished_flags = state("finishedFlags").asInstanceOf[Tensor[Boolean]]
    var notAtMaxDecodeLength = true
    if (i(Array(1)) < maxDecodeLength) {
      notAtMaxDecodeLength = true
    } else {
      notAtMaxDecodeLength = false
    }
    val maxLengthNorm = lengthNormalization(alpha, maxDecodeLength)
    val bestAliveScores = alive_log_probs.select(2, 1) / maxLengthNorm
    val newFinishedFlags = Tensor[Double](finished_flags.size())
    newFinishedFlags.applyFun(finished_flags, x => boolToFloat(x))
    finished_scores *= newFinishedFlags
    var lowestFinishedScores = finished_scores.min(2)._1
    lowestFinishedScores += (reduceAny(finished_flags) * (-1.0) + 1.0) * inf
    val worstFinishedScoreBetterThanBestAliveScore =
      reduceAll(lowestFinishedScores, bestAliveScores)
    return notAtMaxDecodeLength && (!worstFinishedScoreBetterThanBestAliveScore)
  }

  // Reshapes first two dimensions in to single dimension.
  def flattenBeamDim(tensor: Tensor[T]): Tensor[T] = {
    val shape = tensor.size()
    val newShape = shape.toBuffer
    newShape(0) = shape(0) * shape(1)
    newShape.remove(1)
    return tensor.reshape(newShape.toArray)
  }

  // Reshapes first dimension back to [batch_size, beam_size].
  def unFlattenBeamDim(tensor: Tensor[T], batchSize: Int, beamSize: Int): Tensor[T] = {
    val shape = tensor.size()
    val newShape = shape.toBuffer
    newShape(0) = batchSize
    newShape.insert(1, beamSize)
    return tensor.reshape(newShape.toArray)
  }

  // logits - log(sum(exp(logits)))
  def logProbFromLogits(logits: Tensor[T]): Tensor[Double] = {
    val logits1 = logits.asInstanceOf[Tensor[Double]]
    val shape = logits1.size()
    val getExp = Tensor[Double](shape)
    getExp.applyFun[Double](logits1, x => exp(x))
    val getSumExp = getExp.sum(3)
    val getLogSumExp = Tensor[Double](getSumExp.size())
    getLogSumExp.applyFun[Double](getSumExp, x => log(x))
    return logits1 - getLogSumExp.repeatTensor(Array(1, 1, shape(2)))
  }

  def gatherBeams(nested: List[Any], beamIndices: Tensor[T], batchSize: Int, newBeamSize: Int)
  : List[Any] = {
    val batchPos = Tensor.range(0, batchSize*newBeamSize-1, 1)/(batchSize*newBeamSize)
  }

  def growAliveSeq(state: Map[String, Any]): Map[String, Any] = {
    val i = state("CUR_INDEX").asInstanceOf[Tensor[Int]]
    val aliveSeq = state("ALIVE_SEQ").asInstanceOf[Tensor[T]]
    val aliveLogProbs = state("ALIVE_LOG_PROBS").asInstanceOf[Tensor[Double]]
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
    val topkBeamIndices = topkIndices.asInstanceOf[Tensor[Int]]/vocabSize

  }

  def searchStep(state: Map[String, Any]): Map[String, Any] = {

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
      "finishedFlags" -> finishedFlags)

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
}


object SequenceBeamSearch