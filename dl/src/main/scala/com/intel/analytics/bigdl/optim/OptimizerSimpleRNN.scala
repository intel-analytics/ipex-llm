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

package com.intel.analytics.bigdl.optim

import java.nio.file.{Files, Paths}

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.LogSoftMax
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}
import org.apache.log4j.Logger

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.util.Random

abstract class AbstractOptimizer[T  : ClassTag](
  protected val model: Module[T],
  protected val criterion: Criterion[T])(implicit ev : TensorNumeric[T]) {

  protected var state: Table = T()
  protected var optimMethod: OptimMethod[T] = new SGD[T]()
  protected var endWhen: Trigger = Trigger.maxIteration(30)

  protected var trainDataSet: Array[Seq[Int]] = null
  protected var trainLabelSet: Array[Seq[Int]] = null

  protected var valDataSet: Array[Seq[Int]] = null
  protected var valLabelSet: Array[Seq[Int]] = null

  protected var validationTrigger: Option[Trigger] = None

  def optimize(): Module[T]

  def setValidation(
    trigger: Trigger,
    dataset: Array[Seq[Int]],
    labelset: Array[Seq[Int]])
  : this.type = {
    this.validationTrigger = Some(trigger)
    this.valDataSet = dataset
    this.valLabelSet = labelset
    this
  }

  def setTrain(
    dataset: Array[Seq[Int]],
    labelset: Array[Seq[Int]])
  : this.type = {
    this.trainDataSet = dataset
    this.trainLabelSet = labelset
    this
  }

  def setState(state: Table): this.type = {
    this.state = state
    this
  }

  def setOptimMethod(method : OptimMethod[T]): this.type = {
    this.optimMethod = method
    this
  }

  def setEndWhen(endWhen: Trigger): this.type = {
    this.endWhen = endWhen
    this
  }
}


class OptimizerSimpleRNN[T : ClassTag](
  model: Module[T],
  criterion: Criterion[T],
  dictionaryLength: Int)
  (implicit ev : TensorNumeric[T])
  extends AbstractOptimizer[T](model, criterion){

  val logger = Logger.getLogger(getClass)

  def validate(): Double = {
    var i = 0
    var accLoss = 0.0
    val valNumOfSents = valDataSet.length
    while (i < valNumOfSents) {
      val (input, labels) = convert(
        valDataSet(i),
        valLabelSet(i)
      )
      val output = model.forward(input)
      val _loss = criterion.forward(output, labels)
      accLoss += _loss.asInstanceOf[Float]
      i += 1
    }
    accLoss / valNumOfSents
  }

  def convert(inputArray: Seq[Int], labelArray: Seq[Int]): (Tensor[T], Tensor[T]) = {
    val numOfWords = inputArray.length
    val input = Tensor[T](numOfWords, dictionaryLength)
    val labels = Tensor[T](numOfWords)
    var i = 0
    while (i < numOfWords) {
      input.setValue(i + 1, inputArray(i).toString.toInt + 1, ev.fromType[Float](1.0f))
      labels.setValue(i + 1, ev.fromType[Int](labelArray(i).toString.toInt + 1))
      i += 1
    }
    (input, labels)
  }

  def predict(input: Tensor[T]): Array[Int] = {
    val logSoftMax = LogSoftMax[Float]()
    val _output = model.forward(input).asInstanceOf[Tensor[Float]]
    val output = logSoftMax.forward(_output)

    val outputIndex = output.max(2)._2
      .storage.array
      .map(_.toInt)
    outputIndex
  }

  override def optimize(): Module[T] = {
    var wallClockTime = 0L
    var count = 0
    var totalLoss = 0.0
    var preLoss = Double.PositiveInfinity

    val numOfSents = trainDataSet.length
    val seq = (0 until numOfSents).toList

    optimMethod.clearHistory(state)
    state("epoch") = state.get[Int]("epoch").getOrElse(1)
    state("neval") = state.get[Int]("neval").getOrElse(1)
    val (weights, grad) = model.getParameters()

    while (!endWhen(state)) {
      val shuffledSeq = Random.shuffle(seq)

      var averageLoss: Double = 0.0
      var i = 0
      model.training()
      while (i < numOfSents) {
        val start = System.nanoTime()

        val (input, labels) = convert(
          trainDataSet(shuffledSeq(i)),
          trainLabelSet(shuffledSeq(i)))

        val dataFetchTime = System.nanoTime()

        def feval(x: Tensor[T]): (T, Tensor[T]) = {
          val output = model.forward(input)
          val _loss = criterion.forward(output, labels)
          model.zeroGradParameters()
          val gradOutputTest =
            criterion.backward(model.output, labels)
          model.backward(input, gradOutputTest)
          (_loss, grad)
        }

        val (_, loss) = optimMethod.optimize(feval, weights, state)
        val end = System.nanoTime()
        wallClockTime += end - start
        averageLoss += loss(0).asInstanceOf[Float]

        if (i % 1000 == 0) {
          logger.info(s"loss is ${averageLoss/(i + 1)}, iteration ${i} time is ${(end - start) / 1e9}, " +
            s"data fetch time is ${(dataFetchTime - start) / 1e9}, " +
            s"train time is ${(end - dataFetchTime) / 1e9}s. " +
            s"Throughput is ${1.0 / (end - start) * 1e9} sents / second")
        }
        state("neval") = state[Int]("neval") + 1
        i += 1
      }
      averageLoss /= numOfSents
      model.evaluate()
      val valLoss = validate()
      logger.info(s"epoch = ${state("epoch")}, Training: Loss = ${averageLoss}")
      logger.info(s"epoch = ${state("epoch")}, Testing: Loss = ${valLoss}")

      if (state("epoch").asInstanceOf[Int] % 2 == 0) {
        val valSeq = Random.shuffle((1 to valDataSet.length).toList).take(10)
        valSeq.foreach(index => {
          val (sampleInput, sampleLabel) = convert(valDataSet(index), valLabelSet(index))
          val sampleOutput = predict(sampleInput)
          logger.info(s"${index}-th, sampleInput = ${valDataSet(index).mkString(",")}")
          logger.info(s"${index}-th, sampleLabel = ${valLabelSet(index).mkString(",")}")
          logger.info(s"${index}-th, sampleOutput = ${sampleOutput.mkString(",")}")
        })
      }
      state("epoch") = state[Int]("epoch") + 1
    }
  model
  }

}
